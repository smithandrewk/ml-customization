import os
from sympy import hyper
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

from lib.utils_simple import *
import argparse
import json
from time import time

from lib.utils_simple import plot_loss_and_f1

argparser = argparse.ArgumentParser()
argparser.add_argument('--fold', type=int, required=True, help='Fold index for leave-one-participant-out cross-validation')
argparser.add_argument('--device', type=int, required=True, default=0, help='GPU device index')
argparser.add_argument('--batch_size', type=int, required=True, default=64, help='batch size')
argparser.add_argument('--model', type=str, default='medium', choices=['simple', 'medium', 'full', 'test'],
                      help='Model architecture: simple (SimpleSmokingCNN), medium (MediumSmokingCNN), full (SmokingCNN)')
argparser.add_argument('--use_augmentation', action='store_true', help='Enable data augmentation')
argparser.add_argument('--jitter_std', type=float, default=0.005, help='Standard deviation for jitter noise')
argparser.add_argument('--magnitude_range', type=float, nargs=2, default=[0.98, 1.02], help='Range for magnitude scaling')
argparser.add_argument('--aug_prob', type=float, default=0.3, help='Probability of applying augmentation')
argparser.add_argument('--prefix', type=str, default='alpha', help='Experiment prefix/directory name')
argparser.add_argument('--early_stopping_patience', type=int, default=40, help='Early stopping patience for base phase')
argparser.add_argument('--early_stopping_patience_target', type=int, default=40, help='Early stopping patience for target phase')
argparser.add_argument('--mode', type=str, default='full_fine_tuning', choices=['full_fine_tuning', 'target_only', 'target_only_fine_tuning'], help='Mode')
argparser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
argparser.add_argument('--target_data_pct', type=float, default=1.0, help='Percentage of target training data to use (0.0-1.0)')
argparser.add_argument('--participants', type=str, nargs='+', default=['tonmoy','asfik','ejaz'], help='List of participant names for cross-validation')
argparser.add_argument('--window_size', type=int, default=3000, help='Window size in samples (e.g., 3000 = 60s at 50Hz)')
argparser.add_argument('--data_path', type=str, default='data/001_test', help='Path to dataset directory')
argparser.add_argument('--n_base_participants', type=str, default='all', help='Number of base participants to use (integer or "all")')
args = argparser.parse_args()

hyperparameters = {
    'fold': args.fold,
    'device':f'cuda:{args.device}',
    'lr': args.lr,
    'batch_size': args.batch_size,
    'early_stopping_patience': args.early_stopping_patience,
    'early_stopping_patience_target': args.early_stopping_patience_target,
    'window_size': args.window_size,
    'participants': args.participants,
    'experiment_prefix': args.prefix,
    'target_participant': None,
    'data_path': args.data_path,
    'model_type': args.model,
    'use_augmentation': args.use_augmentation,
    'jitter_std': args.jitter_std,
    'magnitude_range': args.magnitude_range,
    'aug_prob': args.aug_prob,
    'mode': args.mode,
    'target_data_pct': args.target_data_pct,
    'n_base_participants': args.n_base_participants if args.n_base_participants == 'all' else int(args.n_base_participants)
    }

fold = hyperparameters['fold']
device = hyperparameters['device']
batch_size = hyperparameters['batch_size']
window_size = hyperparameters['window_size']
experiment_prefix = hyperparameters['experiment_prefix']
data_path = hyperparameters['data_path']
participants = hyperparameters['participants']
target_participant = participants[fold]
hyperparameters['target_participant'] = target_participant

new_exp_dir = f'experiments/{experiment_prefix}/fold{fold}_{target_participant}'
os.makedirs(new_exp_dir, exist_ok=False)

if hyperparameters['mode'] == 'target_only':
    print(f"Target-only mode: training only on {target_participant} data.")
    participants = []
else:
    print(f"Leave-one-participant-out mode: using {target_participant} as target participant.")
    participants.remove(target_participant)

    # Apply n_base_participants constraint
    n_base = hyperparameters['n_base_participants']
    if n_base != 'all':
        if n_base > len(participants):
            raise ValueError(f"n_base_participants ({n_base}) cannot exceed available base participants ({len(participants)})")
        participants = participants[:n_base]
        print(f"Using {n_base} base participants: {participants}")

if participants: 
    base_train_dataset = ConcatDataset([TensorDataset(*torch.load(f'{data_path}/{p}_{s}.pt')) for p in participants for s in ['train', 'val']])
    base_val_dataset = ConcatDataset([TensorDataset(*torch.load(f'{data_path}/{p}_test.pt')) for p in participants])

target_train_dataset = TensorDataset(*torch.load(f'{data_path}/{target_participant}_train.pt'))
target_val_dataset = TensorDataset(*torch.load(f'{data_path}/{target_participant}_val.pt'))

model_type = hyperparameters['model_type']
from lib.models import TestModel
model = TestModel()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparameters['lr'])
print(f'Using {model_type} model: {model.__class__.__name__}')

augmenter = None
if hyperparameters['use_augmentation']:
    from lib.utils import TimeSeriesAugmenter
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

if hyperparameters['mode'] == 'target_only':
    phase = 'target'
else:
    phase = 'base'


from lib.train_utils import random_subsample
from lib.train_utils import append_losses_and_f1

if participants:
    print(f'Base train dataset size: {len(base_train_dataset)}')
    base_train_dataset = random_subsample(base_train_dataset, 1)
    print(f'Base train dataset size: {len(base_train_dataset)}')

    print(f'Base val dataset size: {len(base_val_dataset)}')
    base_val_dataset = random_subsample(base_val_dataset, .5)
    print(f'Base val dataset size: {len(base_val_dataset)}')

    base_trainloader = DataLoader(base_train_dataset, batch_size=batch_size, shuffle=True)
    base_valloader = DataLoader(base_val_dataset, batch_size=batch_size, shuffle=False)

# Subsample target training data if specified
target_data_pct = hyperparameters['target_data_pct']
if target_data_pct < 1.0:
    print(f'Target train dataset size: {len(target_train_dataset)}')
    target_train_dataset = random_subsample(target_train_dataset, target_data_pct)
    print(f'Target train dataset size: {len(target_train_dataset)}')

print(f'Target val dataset size: {len(target_val_dataset)}')
target_val_dataset = random_subsample(target_val_dataset, .1)
print(f'Target val dataset size: {len(target_val_dataset)}')


target_trainloader = DataLoader(target_train_dataset, batch_size=batch_size, shuffle=True)
target_valloader = DataLoader(target_val_dataset, batch_size=batch_size, shuffle=False)

print(f'Target train dataset size: {len(target_train_dataset)}')
print(f'Target val dataset size: {len(target_val_dataset)}')

if hyperparameters['mode'] == 'target_only':
    trainloader = target_trainloader
    valloader = target_valloader
else:
    trainloader = base_trainloader
    valloader = base_valloader

while True:
    if epoch >= 500:
        print("Maximum epochs reached.")
        torch.save(model.state_dict(), f'{new_exp_dir}/last_{phase}_model.pt')
        break

    start_time = time()
    model.train()
    train_loss, train_f1 = optimize_model_compute_loss_and_f1(model, trainloader, optimizer, criterion, device=device, augmenter=augmenter)
    lossi = append_losses_and_f1(phase, train_loss, train_f1, lossi, hyperparameters)

    # loss,f1 = compute_loss_and_f1(model, target_trainloader, criterion, device=device)
    # lossi['target train loss'].append(loss)
    # lossi['target train f1'].append(f1)

    loss,f1 = compute_loss_and_f1(model, valloader, criterion, device=device)

    # In target_only mode, base_valloader is actually target val data
    if hyperparameters['mode'] == 'target_only':
        lossi['target val loss'].append(loss)
        lossi['target val f1'].append(f1)
    else:
        lossi['base val loss'].append(loss)
        lossi['base val f1'].append(f1)

        # Only compute target val loss separately when not in target_only mode
        loss,f1 = compute_loss_and_f1(model, target_valloader, criterion, device=device)
        lossi['target val loss'].append(loss)
        lossi['target val f1'].append(f1)

    # For target_only modes, use target val loss for early stopping; otherwise use phase-specific val loss
    if hyperparameters['mode'] in ['target_only']:
        val_loss_key = 'target val loss'
        val_f1_key = 'target val f1'
    else:
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
        if phase == 'base' and hyperparameters['mode'] not in ['generic', 'target_only']:
            torch.save(model.state_dict(), f'{new_exp_dir}/last_{phase}_model.pt')
            phase = 'target'
            if hyperparameters['mode'] == 'target_only_fine_tuning':
                trainloader = DataLoader(target_train_dataset, batch_size=batch_size, shuffle=True)
            else:
                trainloader = DataLoader(ConcatDataset([base_train_dataset, target_train_dataset]), batch_size=batch_size, shuffle=True)
            best_val_loss = float('inf')
            patience_counter = 0

            metrics['transition_epoch'] = epoch
        else:
            print("Early stopping triggered.")
            torch.save(model.state_dict(), f'{new_exp_dir}/last_{phase}_model.pt')
            break

    if epoch % 10 == 0:
        plot_loss_and_f1(lossi, new_exp_dir, metrics, patience_counter)

    epoch += 1
    # print(f'Epoch {epoch}, Phase: {phase}, Time Elapsed: {time() - start_time:.2f}s, Patience Counter: {patience_counter}, Train F1: {train_f1:.4f}, Val Loss: {lossi[val_loss_key][-1]:.4f}, Val F1: {lossi[val_f1_key][-1]:.4f}')
    print(f'Epoch {epoch}, Phase: {phase}, Time Elapsed: {time() - start_time:.2f}s, Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}')


# # Evaluate best models on target test set
# target_testloader = DataLoader(TensorDataset(*torch.load(f'{data_path}/{target_participant}_test.pt')), batch_size=batch_size)
# # Evaluate best base model on target test set
# if hyperparameters['mode'] == 'full_fine_tuning':
#     model.load_state_dict(torch.load(f'{new_exp_dir}/best_base_model.pt'))
#     model.to(device)
#     test_loss, test_f1 = compute_loss_and_f1(model, target_testloader, criterion, device=device)
#     metrics['best_base_model_target_test_loss'] = test_loss
#     metrics['best_base_model_target_test_f1'] = test_f1

# # Evaluate best target model on target test set
# model.load_state_dict(torch.load(f'{new_exp_dir}/best_target_model.pt'))
# model.to(device)
# test_loss, test_f1 = compute_loss_and_f1(model, target_testloader, criterion, device=device)
# metrics['best_target_model_target_test_loss'] = test_loss
# metrics['best_target_model_target_test_f1'] = test_f1

# # Also evaluate best base model (from base phase) on target val set
# if hyperparameters['mode'] == 'full_fine_tuning':
#     metrics['best_base_model_target_val_loss'] = lossi['target val loss'][metrics['best_base_val_loss_epoch']]
#     metrics['best_base_model_target_val_f1'] = lossi['target val f1'][metrics['best_base_val_loss_epoch']]

# metrics['best_target_model_target_val_loss'] = lossi['target val loss'][metrics['best_target_val_loss_epoch']]
# metrics['best_target_model_target_val_f1'] = lossi['target val f1'][metrics['best_target_val_loss_epoch']]

from lib.train_utils import save_metrics_and_losses
plot_loss_and_f1(lossi, new_exp_dir, metrics, patience_counter)
save_metrics_and_losses(metrics, lossi, hyperparameters, new_exp_dir)