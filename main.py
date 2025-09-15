import os
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

from utils_simple import *
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--fold', type=int, required=True, help='Fold index for leave-one-participant-out cross-validation')
args = argparser.parse_args()
fold = args.fold

# Hyperparameters
lr = 3e-4
batch_size = 32
early_stopping_patience = 5
early_stopping_patience_target = 5
window_size = 3000

data_path = f'data/001_test'
participants = ['alsaad','anam','asfik','ejaz','iftakhar','tonmoy','unk1','dennis']
target_participant = participants[fold]

new_exp_dir = create_and_get_new_exp_dir(prefix='alpha')
participants.remove(target_participant)

base_train_dataset = ConcatDataset([TensorDataset(*torch.load(f'{data_path}/{p}_{s}.pt')) for p in participants for s in ['train', 'val']])
base_val_dataset = ConcatDataset([TensorDataset(*torch.load(f'{data_path}/{p}_test.pt')) for p in participants])

target_train_dataset = TensorDataset(*torch.load(f'{data_path}/{target_participant}_train.pt'))
target_val_dataset = TensorDataset(*torch.load(f'{data_path}/{target_participant}_val.pt'))
target_test_dataset = TensorDataset(*torch.load(f'{data_path}/{target_participant}_test.pt'))

base_valloader = DataLoader(base_val_dataset, batch_size=batch_size)
target_trainloader = DataLoader(target_train_dataset, batch_size=batch_size)
target_valloader = DataLoader(target_val_dataset, batch_size=batch_size)

# Print dataset sizes
print(f'Base train dataset size: {len(base_train_dataset)}')
print(f'Base val dataset size: {len(base_val_dataset)}')
print(f'Target train dataset size: {len(target_train_dataset)}')
print(f'Target val dataset size: {len(target_val_dataset)}')
print(f'Target test dataset size: {len(target_test_dataset)}')

# from utils import SmokingCNN
from utils import SimpleSmokingCNN
model = SimpleSmokingCNN(window_size=window_size, num_features=6)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# Metrics
metrics = {
    'transition_epoch': None,
    'best_base_val_loss': None,
    'best_base_val_loss_epoch': None,
    'best_target_val_loss_epoch': None,
    'best_target_val_loss_epoch': None,
}
best_val_loss = float('inf')
patience_counter = 0
phase = 'base'
device = 'cuda'

evalloaders = [
    (base_valloader, 'base val', 'b'), 
    (target_trainloader, 'target train', 'g'),
    (target_valloader, 'target val', 'g')
    ]

lossi = {}
for evalloader in evalloaders:
    lossi[f'{evalloader[1]} loss'] = []
    lossi[f'{evalloader[1]} f1'] = []
lossi['train loss'] = []
lossi['train f1'] = []

trainloader = DataLoader(base_train_dataset, batch_size=batch_size, shuffle=True)
model.to(device)
epoch = 0

from time import time

while True:
    start_time = time()
    model.train()
    train_loss, train_f1 = optimize_model_compute_loss_and_f1(model, trainloader, optimizer, criterion, device=device)

    lossi['train loss'].append(train_loss)
    lossi['train f1'].append(train_f1)

    for evalloader in evalloaders:
        valloader, name, color = evalloader
        val_loss, val_f1 = compute_loss_and_f1(model, valloader, criterion, device=device)
        lossi[f'{name} loss'].append(val_loss)
        lossi[f'{name} f1'].append(val_f1)

    if lossi[f'{phase} val loss'][-1] < best_val_loss:
        best_val_loss = lossi[f'{phase} val loss'][-1]
        metrics[f'best_{phase}_val_loss'] = best_val_loss
        metrics[f'best_{phase}_val_loss_epoch'] = epoch
        torch.save(model.state_dict(), f'{new_exp_dir}/best_{phase}_model.pt')
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= early_stopping_patience:
        if phase == 'base':
            print("Switching to target phase")
            torch.save(model.state_dict(), f'{new_exp_dir}/last_{phase}_model.pt')
            phase = 'target'
            trainloader = DataLoader(ConcatDataset([base_train_dataset, target_train_dataset]), batch_size=batch_size, shuffle=True)
            best_val_loss = float('inf')
            patience_counter = 0
            early_stopping_patience = early_stopping_patience_target
            metrics['transition_epoch'] = epoch
        else:
            print("Early stopping triggered")
            torch.save(model.state_dict(), f'{new_exp_dir}/last_{phase}_model.pt')
            break

    plt.figure(figsize=(7.2,4.48))

    plt.plot(lossi['train loss'], label='Train Loss (base)')

    for evalloader in evalloaders:
        name = evalloader[1]
        plt.plot(lossi[f'{name} loss'], label=f'{name} loss', color=evalloader[2], linestyle='--')

    if metrics['transition_epoch'] is not None:
        plt.axvline(x=metrics['transition_epoch'], color='r', linestyle='--', label='Phase Transition')

    if metrics['best_base_val_loss_epoch'] is not None and metrics['best_base_val_loss'] is not None:
        plt.axhline(y=metrics['best_base_val_loss'], color='b', linestyle='--', label='Best Base Val Loss', alpha=0.5)
        plt.axvline(x=metrics['best_base_val_loss_epoch'], color='b', linestyle='--', alpha=0.5)
        plt.axhline(y=lossi['target val loss'][metrics['best_base_val_loss_epoch']], color='g', linestyle='--', label='Best Base Val Loss', alpha=0.5)

    if metrics['best_target_val_loss_epoch'] is not None and metrics['best_target_val_loss'] is not None:
        plt.axhline(y=metrics['best_target_val_loss'], color='g', linestyle='--', label='Best target Val Loss', alpha=0.5)
        plt.axvline(x=metrics['best_target_val_loss_epoch'], color='g', linestyle='--', alpha=0.5)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.yscale('log')
    plt.title(f'Patience Counter: {patience_counter}')
    plt.savefig(f'{new_exp_dir}/loss.jpg', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(7.2,4.48))
    plt.plot(lossi['train f1'], label='Train f1 (base)', color='b')
    plt.plot(lossi['base val f1'], label='Val f1 (base)', color='b', linestyle='--')
    plt.plot(lossi['target train f1'], label='Train f1 (target)', color='r')
    plt.plot(lossi['target val f1'], label='Val f1 (target)', color='r', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.ylim([.7, 1])
    plt.legend()
    plt.title(f'Patience Counter: {patience_counter}')
    plt.savefig(f'{new_exp_dir}/f1.jpg', bbox_inches='tight')
    plt.close()

    epoch += 1
    print(f'Epoch {epoch}, Phase: {phase}, Time Elapsed: {time() - start_time:.2f}s')
