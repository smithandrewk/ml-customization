import os
from sympy import hyper
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

from lib.utils_simple import *
import argparse
import json
from time import time

argparser = argparse.ArgumentParser()
argparser.add_argument('--fold', type=int, required=True, help='Fold index for leave-one-participant-out cross-validation')
argparser.add_argument('--device', type=int, required=True, default=0, help='GPU device index')
argparser.add_argument('-b','--batch_size', type=int, required=True, default=64, help='batch size')
argparser.add_argument('--model', type=str, default='medium', choices=['simple', 'medium', 'full'],
                      help='Model architecture: simple (SimpleSmokingCNN), medium (MediumSmokingCNN), full (SmokingCNN)')
argparser.add_argument('--use_augmentation', action='store_true', help='Enable data augmentation')
argparser.add_argument('--jitter_std', type=float, default=0.005, help='Standard deviation for jitter noise')
argparser.add_argument('--magnitude_range', type=float, nargs=2, default=[0.98, 1.02], help='Range for magnitude scaling')
argparser.add_argument('--aug_prob', type=float, default=0.3, help='Probability of applying augmentation')
argparser.add_argument('--prefix', type=str, default='alpha', help='Experiment prefix/directory name')
argparser.add_argument('--early_stopping_patience', type=int, default=40, help='Early stopping patience for base phase')
argparser.add_argument('--early_stopping_patience_target', type=int, default=40, help='Early stopping patience for target phase')
argparser.add_argument('--mode', type=str, default='full_fine_tuning', choices=['full_fine_tuning', 'last_layer_only', 'generic', 'target_only'], help='Mode')
argparser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
args = argparser.parse_args()

hyperparameters = {
    'fold': args.fold,
    'device':f'cuda:{args.device}',
    'lr': args.lr,
    'batch_size': args.batch_size,
    'early_stopping_patience': args.early_stopping_patience,
    'early_stopping_patience_target': args.early_stopping_patience_target,
    'window_size': 3000,
    'participants': ['alsaad','anam','asfik','ejaz','iftakhar','tonmoy','unk1','dennis'],
    'experiment_prefix': args.prefix + f'_{args.mode}',
    'target_participant': None,  # to be set later
    'data_path': 'data/001_test',
    'model_type': args.model,
    'use_augmentation': args.use_augmentation,
    'jitter_std': args.jitter_std,
    'magnitude_range': args.magnitude_range,
    'aug_prob': args.aug_prob,
    'mode': args.mode
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

new_exp_dir = create_and_get_new_exp_dir(prefix=experiment_prefix)

if hyperparameters['mode'] == 'generic':
    print("Generic mode: using ALL participants including target in training data.")
    # Keep all participants - don't remove target
elif hyperparameters['mode'] == 'target_only':
    print(f"Target-only mode: training only on {target_participant} data.")
    participants = []  # Empty - no other participants
else:
    print(f"Leave-one-participant-out mode: using {target_participant} as target participant.")
    participants.remove(target_participant)

if participants:  # Not empty (generic or personalization modes)
    base_train_dataset = ConcatDataset([TensorDataset(*torch.load(f'{data_path}/{p}_{s}.pt')) for p in participants for s in ['train', 'val']])
    base_val_dataset = ConcatDataset([TensorDataset(*torch.load(f'{data_path}/{p}_test.pt')) for p in participants])
else:  # target_only mode
    # Use target data as "base" data (we'll only train on this)
    base_train_dataset = TensorDataset(*torch.load(f'{data_path}/{target_participant}_train.pt'))
    base_val_dataset = TensorDataset(*torch.load(f'{data_path}/{target_participant}_val.pt'))

target_train_dataset = TensorDataset(*torch.load(f'{data_path}/{target_participant}_train.pt'))
target_val_dataset = TensorDataset(*torch.load(f'{data_path}/{target_participant}_val.pt'))
target_test_dataset = TensorDataset(*torch.load(f'{data_path}/{target_participant}_test.pt'))

trainloader = DataLoader(base_train_dataset, batch_size=batch_size, shuffle=True)
base_valloader = DataLoader(base_val_dataset, batch_size=batch_size)
target_trainloader = DataLoader(target_train_dataset, batch_size=batch_size)
target_valloader = DataLoader(target_val_dataset, batch_size=batch_size)

print(f'Base train dataset size: {len(base_train_dataset)}')
print(f'Base val dataset size: {len(base_val_dataset)}')
print(f'Target train dataset size: {len(target_train_dataset)}')
print(f'Target val dataset size: {len(target_val_dataset)}')
print(f'Target test dataset size: {len(target_test_dataset)}')

from lib.utils import SimpleSmokingCNN, MediumSmokingCNN, SmokingCNN, calculate_receptive_field

model_type = hyperparameters['model_type']
if model_type == 'simple':
    model = SimpleSmokingCNN(window_size=window_size, num_features=6)
elif model_type == 'medium':
    model = MediumSmokingCNN(window_size=window_size, num_features=6)
elif model_type == 'full':
    model = SmokingCNN(window_size=window_size, num_features=6)
else:
    raise ValueError(f"Invalid model type: {model_type}. Choose from 'simple', 'medium', 'full'")
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparameters['lr'])

receptive_field = calculate_receptive_field(model)
print(f'Using {model_type} model: {model.__class__.__name__}')
print(f'Receptive field: {receptive_field} samples ({receptive_field/50:.1f}s at 50Hz)')

augmenter = None
if hyperparameters['use_augmentation']:
    from lib.utils import TimeSeriesAugmenter
    augmenter = TimeSeriesAugmenter({
        'jitter_noise_std': hyperparameters['jitter_std'],
        'magnitude_scale_range': hyperparameters['magnitude_range'],
        'augmentation_probability': hyperparameters['aug_prob']
    })
    print(f'Data augmentation enabled: jitter_std={hyperparameters["jitter_std"]}, mag_range={hyperparameters["magnitude_range"]}, prob={hyperparameters["aug_prob"]}')

print(f'Total model parameters: {sum(p.numel() for p in model.parameters())}')

metrics = {
    'receptive_field': receptive_field,
    'receptive_field_seconds': receptive_field / 50.0,
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
phase = 'base'

while True:
    start_time = time()
    model.train()
    train_loss, train_f1 = optimize_model_compute_loss_and_f1(model, trainloader, optimizer, criterion, device=device, augmenter=augmenter)

    lossi['base train loss'].append(train_loss)
    lossi['base train f1'].append(train_f1)

    loss,f1 = compute_loss_and_f1(model, base_valloader, criterion, device=device)
    lossi['base val loss'].append(loss)
    lossi['base val f1'].append(f1)

    loss,f1 = compute_loss_and_f1(model, target_trainloader, criterion, device=device)
    lossi['target train loss'].append(loss)
    lossi['target train f1'].append(f1)

    loss,f1 = compute_loss_and_f1(model, target_valloader, criterion, device=device)
    lossi['target val loss'].append(loss)
    lossi['target val f1'].append(f1)

    if lossi[f'{phase} val loss'][-1] < best_val_loss:
        best_val_loss = lossi[f'{phase} val loss'][-1]
        metrics[f'best_{phase}_val_loss'] = best_val_loss
        metrics[f'best_{phase}_val_loss_epoch'] = epoch
        torch.save(model.state_dict(), f'{new_exp_dir}/best_{phase}_model.pt')
        patience_counter = 0
    else:
        patience_counter += 1

    if lossi[f'{phase} val f1'][-1] > (metrics[f'best_{phase}_val_f1'] or 0):
        metrics[f'best_{phase}_val_f1'] = lossi[f'{phase} val f1'][-1]
        metrics[f'best_{phase}_val_f1_epoch'] = epoch

    if patience_counter >= hyperparameters['early_stopping_patience' if phase == 'base' else 'early_stopping_patience_target']:
        if phase == 'base' and hyperparameters['mode'] not in ['generic', 'target_only']:
            torch.save(model.state_dict(), f'{new_exp_dir}/last_{phase}_model.pt')
            phase = 'target'
            trainloader = DataLoader(ConcatDataset([base_train_dataset, target_train_dataset]), batch_size=batch_size, shuffle=True)
            best_val_loss = float('inf')
            patience_counter = 0
            
            metrics['transition_epoch'] = epoch
            metrics['best_target_val_loss_from_best_base_model'] = lossi['target val loss'][metrics['best_base_val_loss_epoch']]
            metrics['best_target_val_f1_from_best_base_model'] = lossi['target val f1'][metrics['best_base_val_loss_epoch']]

            if hyperparameters['mode'] == 'last_layer_only':
                for param in model.parameters():
                    param.requires_grad = False
                if hasattr(model, 'fc'):
                    for param in model.fc.parameters():
                        param.requires_grad = True
                elif hasattr(model, 'classifier'):
                    for param in model.classifier.parameters():
                        param.requires_grad = True
                else:
                    raise ValueError("Model does not have 'fc' or 'classifier' attribute for last layer fine-tuning.")
                print("Fine-tuning only the last layer of the model.")
                print(f'Trainable model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
        else:
            print("Early stopping triggered.")
            torch.save(model.state_dict(), f'{new_exp_dir}/last_{phase}_model.pt')

            with open(f'{new_exp_dir}/metrics.json', 'w') as f:
                json.dump(metrics, f, indent=4)
            with open(f'{new_exp_dir}/losses.json', 'w') as f:
                json.dump(lossi, f, indent=4)
            with open(f'{new_exp_dir}/hyperparameters.json', 'w') as f:
                json.dump(hyperparameters, f, indent=4)
            break

    plt.figure(figsize=(7.2,4.48))
    plt.plot(lossi['base train loss'], label='Train Loss (base)', color='b')
    plt.plot(lossi['base val loss'], label='Val Loss (base)', color='b', linestyle='--')
    plt.plot(lossi['target train loss'], label='Train Loss (target)', color='g')
    plt.plot(lossi['target val loss'], label='Val Loss (target)', color='g', linestyle='--')

    if metrics['transition_epoch'] is not None:
        plt.axvline(x=metrics['transition_epoch'], color='r', linestyle='--', label='Phase Transition')

    if metrics['best_base_val_loss_epoch'] is not None and metrics['best_base_val_loss'] is not None:
        plt.axhline(y=metrics['best_base_val_loss'], color='b', linestyle='--', label='Best Base Val Loss', alpha=0.5)
        plt.axvline(x=metrics['best_base_val_loss_epoch'], color='b', linestyle='--', alpha=0.5)
        plt.axhline(y=lossi['target val loss'][metrics['best_base_val_loss_epoch']], color='g', linestyle='--', label='Best Base Val Loss', alpha=0.5)

    if metrics['best_target_val_loss_epoch'] is not None and metrics['best_target_val_loss'] is not None:
        plt.axhline(y=metrics['best_target_val_loss'], color='g', linestyle='--', label='Best target Val Loss', alpha=0.8)
        plt.axvline(x=metrics['best_target_val_loss_epoch'], color='g', linestyle='--', alpha=0.4)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.yscale('log')
    plt.title(f'Patience Counter: {patience_counter}')
    plt.savefig(f'{new_exp_dir}/loss.jpg', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(7.2,4.48))
    plt.plot(lossi['base train f1'], label='Train f1 (base)', color='b')
    plt.plot(lossi['base val f1'], label='Val f1 (base)', color='b', linestyle='--')
    plt.plot(lossi['target train f1'], label='Train f1 (target)', color='g')
    plt.plot(lossi['target val f1'], label='Val f1 (target)', color='g', linestyle='--')

    if metrics['transition_epoch'] is not None:
        plt.axvline(x=metrics['transition_epoch'], color='r', linestyle='--', label='Phase Transition')

    if metrics['best_base_val_f1_epoch'] is not None and metrics['best_base_val_f1'] is not None:
        plt.axhline(y=metrics['best_base_val_f1'], color='b', linestyle='--', label='Best Base f1 Loss', alpha=0.5)
        plt.axvline(x=metrics['best_base_val_f1_epoch'], color='b', linestyle='--', alpha=0.5)
        plt.axhline(y=lossi['target val f1'][metrics['best_base_val_f1_epoch']], color='g', linestyle='--', label='Best Base Val Loss', alpha=0.5)

    if metrics['best_target_val_f1_epoch'] is not None and metrics['best_target_val_f1'] is not None:
        plt.axhline(y=metrics['best_target_val_f1'], color='g', linestyle='--', label='Best target f1 Loss', alpha=0.8)
        plt.axvline(x=metrics['best_target_val_f1_epoch'], color='g', linestyle='--', alpha=0.4)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Patience Counter: {patience_counter}')
    plt.savefig(f'{new_exp_dir}/f1.jpg', bbox_inches='tight')
    plt.close()

    epoch += 1
    print(f'Epoch {epoch}, Phase: {phase}, Time Elapsed: {time() - start_time:.2f}s')


