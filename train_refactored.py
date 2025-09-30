import argparse
import torch
from lib.models import *
from time import time
import os
from lib.utils_simple import optimize_model_compute_loss_and_f1, compute_loss_and_f1

parser = argparse.ArgumentParser(description='Create participant-specific smoking detection dataset')
parser.add_argument('--config', type=str, default='configs/train_config.yaml', help='Path to config file')
args = parser.parse_args()

from lib.train_utils import load_config, load_data, save_metrics_and_losses,plot_loss_and_f1_refactored
config = load_config(args.config)

training_mode = config['training'].get('mode', 'target_only')
device = 'cuda'
plotting_frequency_epochs = config['visualization'].get('plotting_frequency_epochs', 5)
use_lr_scheduler = config['training'].get('lr_scheduler', {}).get('use_scheduler', False)
lr_scheduler_patience = config['training'].get('lr_scheduler', {}).get('patience', 10)
use_early_stopping = config['training'].get('early_stopping', {}).get('use_early_stopping', False)
early_stopping_patience = config['training'].get('early_stopping', {}).get('patience', 50)

target_trainloader, target_valloader = load_data(config)

if training_mode == 'target_only':
    print("Training on target participant only.")
    trainloader = target_trainloader
    valloader = target_valloader

experiment_name = int(time())
new_exp_dir = f'experiments/{experiment_name}'
os.makedirs(new_exp_dir, exist_ok=False)

model = TestModel()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['lr']))
if use_lr_scheduler:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=30)

from lib.utils_simple import EarlyStopper
if use_early_stopping:
    early_stopper = EarlyStopper(patience=early_stopping_patience)

print(f'Total model parameters: {sum(p.numel() for p in model.parameters())}')

metrics = {
    'transition_epoch': None,
    'best_base_val_loss': None,
    'best_base_val_loss_epoch': None,
    'best_target_val_loss_epoch': None,
    'best_target_val_loss_epoch': None,
    'best_base_val_f1': None,
    'best_base_val_f1_epoch': None,
    'best_target_val_f1': None,
    'best_target_val_f1_epoch': None,
}

lossi = {
    'base train loss': [],
    'base train f1': [],
    'base val loss': [],
    'base val f1': [],
    'target train loss': [],
    'target train f1': [],
    'target val loss': [],
    'target val f1': [],
}
model.to(device)
epoch = 0
best_val_loss = float('inf')
patience_counter = 0

while True:
    start_time = time()
    model.train()
    train_loss, train_f1 = optimize_model_compute_loss_and_f1(model, trainloader, optimizer, criterion, device=device, augmenter=None)

    lossi['target train loss'].append(train_loss)
    lossi['target train f1'].append(train_f1)

    loss,f1 = compute_loss_and_f1(model, target_valloader, criterion, device=device)
    lossi['target val loss'].append(loss)
    lossi['target val f1'].append(f1)

    if use_lr_scheduler:
        scheduler.step(loss)

    if use_early_stopping:
        if early_stopper.step(loss):
            print(f"Early stopping triggered at epoch {epoch}. Best val loss: {early_stopper.best_loss:.4f}")

            save_metrics_and_losses(metrics, lossi, config, new_exp_dir)
            break
        if early_stopper.counter == 0:
            torch.save(model.state_dict(), f'{new_exp_dir}/best_model.pt')

    if plotting_frequency_epochs and epoch % plotting_frequency_epochs == 0:
        plot_loss_and_f1_refactored(lossi, new_exp_dir)

    epoch += 1
    print(f'Epoch {epoch}, Time Elapsed: {time() - start_time:.2f}s, Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}, Val Loss: {loss:.4f}, Val F1: {f1:.4f}, LR: {optimizer.param_groups[0]["lr"]:.2e} Patience Counter: {early_stopper.counter}')