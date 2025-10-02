import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import os
import argparse
from time import time
from lib.train_utils import *

argparser = argparse.ArgumentParser()
argparser = add_arguments(argparser)
args = argparser.parse_args()

# Check that data_path exists
if not os.path.exists(args.data_path):
    raise ValueError(f"Data path {args.data_path} does not exist.")

hyperparameters = vars(args)

fold = hyperparameters['fold']
device = hyperparameters['device']
batch_size = hyperparameters['batch_size']
window_size = hyperparameters['window_size']
experiment_prefix = hyperparameters['prefix']
data_path = hyperparameters['data_path']
participants = hyperparameters['participants']
target_participant = participants[fold]
hyperparameters['target_participant'] = target_participant

new_exp_dir = f'experiments/{experiment_prefix}/fold{fold}_{target_participant}'
os.makedirs(new_exp_dir, exist_ok=False)

print(f"Leave-one-participant-out mode: using {target_participant} as target participant.")
participants.remove(target_participant)

# Apply n_base_participants constraint
n_base = hyperparameters['n_base_participants']
if n_base != 'all':
    n_base = int(n_base)
    if n_base > len(participants):
        raise ValueError(f"n_base_participants ({n_base}) cannot exceed available base participants ({len(participants)})")
    participants = participants[:n_base]
    print(f"Using {n_base} base participants: {participants}")

if participants: 
    base_train_dataset = ConcatDataset([TensorDataset(*torch.load(f'{data_path}/{p}_{s}.pt')) for p in participants for s in ['train', 'val']])
    base_val_dataset = ConcatDataset([TensorDataset(*torch.load(f'{data_path}/{p}_test.pt')) for p in participants])

from lib.models import TestModel
model = TestModel()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparameters['lr'])
print(f'Using model: {model.__class__.__name__}')

augmenter = None
if hyperparameters['use_augmentation']:
    from lib.train_utils import TimeSeriesAugmenter
    augmenter = TimeSeriesAugmenter({
        'jitter_noise_std': hyperparameters['jitter_std'],
        'magnitude_scale_range': hyperparameters['magnitude_range'],
        'augmentation_probability': hyperparameters['aug_prob']
    })

print(f'Total model parameters: {sum(p.numel() for p in model.parameters())}')

metrics = {
    'transition_epoch': None,
    'best_base_val_loss': None,
    'best_base_val_loss_epoch': None,
    'best_base_val_f1': None,
    'best_base_val_f1_epoch': None,
}

lossi = {
    'base train loss': [],
    'base train f1': [],
    'base val loss': [],
    'base val f1': [],
}

model.to(device)
epoch = 0
best_val_loss = float('inf')
patience_counter = 0
phase = 'base'


from lib.train_utils import random_subsample
from lib.train_utils import append_losses_and_f1

base_trainloader = DataLoader(base_train_dataset, batch_size=batch_size, shuffle=True)
base_valloader = DataLoader(base_val_dataset, batch_size=batch_size, shuffle=False)

trainloader = base_trainloader
valloader = base_valloader

while True:
    start_time = time()
    model.train()
    train_loss, train_f1 = optimize_model_compute_loss_and_f1(model, trainloader, optimizer, criterion, device=device, augmenter=augmenter)

    lossi['base train loss'].append(train_loss)
    lossi['base train f1'].append(train_f1)

    val_loss,val_f1 = compute_loss_and_f1(model, valloader, criterion, device=device)

    lossi['base val loss'].append(val_loss)
    lossi['base val f1'].append(val_f1)

    val_loss_key = f'{phase} val loss'
    val_f1_key = f'{phase} val f1'

    # Early Stopping
    if lossi[val_loss_key][-1] < best_val_loss:
        best_val_loss = lossi[val_loss_key][-1]
        metrics[f'best_{phase}_val_loss'] = best_val_loss
        metrics[f'best_{phase}_val_loss_epoch'] = epoch
        torch.save(model.state_dict(), f'{new_exp_dir}/best_{phase}_model.pt')
        patience_counter = 0
    else:
        patience_counter += 1

    if lossi[val_f1_key][-1] > (metrics[f'best_{phase}_val_f1'] or 0):
        metrics[f'best_{phase}_val_f1'] = lossi[val_f1_key][-1]
        metrics[f'best_{phase}_val_f1_epoch'] = epoch

    if patience_counter >= hyperparameters['early_stopping_patience' if phase == 'base' else 'early_stopping_patience_target']:
        torch.save(model.state_dict(), f'{new_exp_dir}/last_{phase}_model.pt')
        metrics['transition_epoch'] = epoch
        print("Early stopping triggered.")
        torch.save(model.state_dict(), f'{new_exp_dir}/last_{phase}_model.pt')
        break

    if epoch % 5 == 0:
        plot_loss_and_f1(lossi, new_exp_dir, metrics, patience_counter)

    epoch += 1
    print(f'Epoch {epoch}, Phase: {phase}, Time Elapsed: {time() - start_time:.2f}s, Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}, Patience Counter: {patience_counter}')

plot_loss_and_f1(lossi, new_exp_dir, metrics, patience_counter)
save_metrics_and_losses(metrics, lossi, hyperparameters, new_exp_dir)