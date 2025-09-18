#!/usr/bin/env python3
"""
Baseline training script for smoking detection models.
Supports multiple training strategies for comparison with personalization approaches.
"""

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
argparser.add_argument('--prefix', type=str, default='baseline', help='Experiment prefix/directory name')
argparser.add_argument('--early_stopping_patience', type=int, default=40, help='Early stopping patience for base phase')
argparser.add_argument('--early_stopping_patience_target', type=int, default=40, help='Early stopping patience for target phase')

# New baseline-specific arguments
argparser.add_argument('--mode', type=str, required=True,
                      choices=['personalized', 'generic', 'last_layer_only', 'target_only'],
                      help='Training mode: personalized (original 2-phase), generic (all data), last_layer_only (freeze base layers), target_only (no base training)')

args = argparser.parse_args()

hyperparameters = {
    'fold': args.fold,
    'device':f'cuda:{args.device}',
    'lr': 3e-4,
    'batch_size': args.batch_size,
    'early_stopping_patience': args.early_stopping_patience,
    'early_stopping_patience_target': args.early_stopping_patience_target,
    'window_size': 3000,
    'participants': ['alsaad','anam','asfik','ejaz','iftakhar','tonmoy','unk1','dennis'],
    'experiment_prefix': args.prefix,
    'target_participant': None,  # to be set later
    'data_path': 'data/001_test',
    'model_type': args.model,
    'use_augmentation': args.use_augmentation,
    'jitter_std': args.jitter_std,
    'magnitude_range': args.magnitude_range,
    'aug_prob': args.aug_prob,
    'training_mode': args.mode  # New parameter to track baseline type
}

fold = hyperparameters['fold']
device = hyperparameters['device']
batch_size = hyperparameters['batch_size']
window_size = hyperparameters['window_size']
experiment_prefix = hyperparameters['experiment_prefix']
data_path = hyperparameters['data_path']
participants = hyperparameters['participants']
target_participant = participants[fold]
training_mode = hyperparameters['training_mode']

new_exp_dir = create_and_get_new_exp_dir(prefix=f"{experiment_prefix}_{training_mode}")

print(f"Training mode: {training_mode}")
print(f"Target participant: {target_participant}")

# Setup datasets based on training mode
if training_mode == 'generic':
    # Generic: Use ALL participants from the start (no leave-one-out)
    all_participants = participants.copy()
    base_train_dataset = ConcatDataset([TensorDataset(*torch.load(f'{data_path}/{p}_{s}.pt'))
                                       for p in all_participants for s in ['train', 'val']])
    base_val_dataset = ConcatDataset([TensorDataset(*torch.load(f'{data_path}/{p}_test.pt'))
                                     for p in all_participants])

    target_train_dataset = TensorDataset(*torch.load(f'{data_path}/{target_participant}_train.pt'))
    target_val_dataset = TensorDataset(*torch.load(f'{data_path}/{target_participant}_val.pt'))
    target_test_dataset = TensorDataset(*torch.load(f'{data_path}/{target_participant}_test.pt'))

    print("Generic mode: Training on ALL participants from start")

elif training_mode == 'target_only':
    # Target-only: Use only target participant's data
    participants.remove(target_participant)

    # Empty base datasets (we'll skip base phase)
    base_train_dataset = None
    base_val_dataset = None

    target_train_dataset = TensorDataset(*torch.load(f'{data_path}/{target_participant}_train.pt'))
    target_val_dataset = TensorDataset(*torch.load(f'{data_path}/{target_participant}_val.pt'))
    target_test_dataset = TensorDataset(*torch.load(f'{data_path}/{target_participant}_test.pt'))

    print("Target-only mode: Training only on target participant data")

else:
    # Personalized and last_layer_only: Use leave-one-out (same as original)
    participants.remove(target_participant)

    base_train_dataset = ConcatDataset([TensorDataset(*torch.load(f'{data_path}/{p}_{s}.pt'))
                                       for p in participants for s in ['train', 'val']])
    base_val_dataset = ConcatDataset([TensorDataset(*torch.load(f'{data_path}/{p}_test.pt'))
                                     for p in participants])

    target_train_dataset = TensorDataset(*torch.load(f'{data_path}/{target_participant}_train.pt'))
    target_val_dataset = TensorDataset(*torch.load(f'{data_path}/{target_participant}_val.pt'))
    target_test_dataset = TensorDataset(*torch.load(f'{data_path}/{target_participant}_test.pt'))

    print(f"{training_mode} mode: Leave-one-out cross-validation")

# Create data loaders
if training_mode == 'target_only':
    # For target-only, we start directly with target data
    trainloader = DataLoader(target_train_dataset, batch_size=batch_size, shuffle=True)
    base_valloader = None  # Not used
else:
    trainloader = DataLoader(base_train_dataset, batch_size=batch_size, shuffle=True)
    base_valloader = DataLoader(base_val_dataset, batch_size=batch_size)

target_trainloader = DataLoader(target_train_dataset, batch_size=batch_size)
target_valloader = DataLoader(target_val_dataset, batch_size=batch_size)

# Print dataset sizes
if training_mode != 'target_only':
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
print(f'Total model parameters: {sum(p.numel() for p in model.parameters())}')
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

metrics = {
    'receptive_field': receptive_field,
    'receptive_field_seconds': receptive_field / 50.0,
    'training_mode': training_mode,
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

# Determine starting phase based on training mode
if training_mode == 'target_only':
    phase = 'target'
    print("Skipping base phase - starting with target phase")
else:
    phase = 'base'

# Training loop
while True:
    start_time = time()
    model.train()

    # Training step
    if phase == 'base' or training_mode == 'generic':
        # Base phase or generic mode: train on base/all data
        train_loss, train_f1 = optimize_model_compute_loss_and_f1(model, trainloader, optimizer, criterion, device=device, augmenter=augmenter)
        lossi['base train loss'].append(train_loss)
        lossi['base train f1'].append(train_f1)

        # For target_only mode, we don't have base validation
        if training_mode != 'target_only' and base_valloader is not None:
            loss, f1 = compute_loss_and_f1(model, base_valloader, criterion, device=device)
            lossi['base val loss'].append(loss)
            lossi['base val f1'].append(f1)
        else:
            # Placeholder values for consistency
            lossi['base val loss'].append(float('nan'))
            lossi['base val f1'].append(float('nan'))

    elif phase == 'target':
        # Target phase: train on target data (or combined for personalized)
        if training_mode == 'personalized':
            # Use combined dataset
            train_loss, train_f1 = optimize_model_compute_loss_and_f1(model, trainloader, optimizer, criterion, device=device, augmenter=augmenter)
        else:
            # Use only target data
            train_loss, train_f1 = optimize_model_compute_loss_and_f1(model, target_trainloader, optimizer, criterion, device=device, augmenter=augmenter)

        lossi['target train loss'].append(train_loss)
        lossi['target train f1'].append(train_f1)

        # Add placeholder base values during target phase
        lossi['base train loss'].append(float('nan'))
        lossi['base train f1'].append(float('nan'))
        if training_mode != 'target_only' and base_valloader is not None:
            loss, f1 = compute_loss_and_f1(model, base_valloader, criterion, device=device)
            lossi['base val loss'].append(loss)
            lossi['base val f1'].append(f1)
        else:
            lossi['base val loss'].append(float('nan'))
            lossi['base val f1'].append(float('nan'))

    # Always evaluate on target validation
    loss, f1 = compute_loss_and_f1(model, target_valloader, criterion, device=device)
    lossi['target val loss'].append(loss)
    lossi['target val f1'].append(f1)

    # Early stopping logic
    current_val_loss = lossi[f'{phase} val loss'][-1]
    if not np.isnan(current_val_loss) and current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        metrics[f'best_{phase}_val_loss'] = best_val_loss
        metrics[f'best_{phase}_val_loss_epoch'] = epoch
        torch.save(model.state_dict(), f'{new_exp_dir}/best_{phase}_model.pt')
        patience_counter = 0
    else:
        patience_counter += 1

    # Update best F1 scores
    current_val_f1 = lossi[f'{phase} val f1'][-1]
    if not np.isnan(current_val_f1) and current_val_f1 > (metrics[f'best_{phase}_val_f1'] or 0):
        metrics[f'best_{phase}_val_f1'] = current_val_f1
        metrics[f'best_{phase}_val_f1_epoch'] = epoch

    # Phase transition or termination logic
    patience_threshold = hyperparameters['early_stopping_patience' if phase == 'base' else 'early_stopping_patience_target']

    if patience_counter >= patience_threshold:
        if phase == 'base' and training_mode in ['personalized', 'last_layer_only']:
            # Transition to target phase
            torch.save(model.state_dict(), f'{new_exp_dir}/last_{phase}_model.pt')
            phase = 'target'

            # Setup for target phase based on mode
            if training_mode == 'personalized':
                # Personalized: use combined dataset
                trainloader = DataLoader(ConcatDataset([base_train_dataset, target_train_dataset]),
                                       batch_size=batch_size, shuffle=True)
            elif training_mode == 'last_layer_only':
                # Last-layer only: freeze all parameters except final layer
                print("Freezing all layers except final classifier...")
                for name, param in model.named_parameters():
                    # Adjust this condition based on your model architecture
                    if 'classifier' not in name and 'fc' not in name and 'output' not in name:
                        param.requires_grad = False
                        print(f"Frozen: {name}")
                    else:
                        print(f"Trainable: {name}")

                # Update optimizer to only include trainable parameters
                trainable_params = [p for p in model.parameters() if p.requires_grad]
                optimizer = torch.optim.AdamW(trainable_params, lr=hyperparameters['lr'])
                print(f"Trainable parameters: {sum(p.numel() for p in trainable_params)}")

                # Use only target data for last-layer fine-tuning
                trainloader = target_trainloader

            best_val_loss = float('inf')
            patience_counter = 0

            metrics['transition_epoch'] = epoch
            if training_mode != 'target_only':
                # Record base model performance on target data
                base_epoch = metrics['best_base_val_loss_epoch']
                if base_epoch is not None and base_epoch < len(lossi['target val loss']):
                    metrics['best_target_val_loss_from_best_base_model'] = lossi['target val loss'][base_epoch]
                    metrics['best_target_val_f1_from_best_base_model'] = lossi['target val f1'][base_epoch]
                else:
                    # Fallback: use current values
                    metrics['best_target_val_loss_from_best_base_model'] = lossi['target val loss'][-1]
                    metrics['best_target_val_f1_from_best_base_model'] = lossi['target val f1'][-1]

        else:
            # Training complete
            torch.save(model.state_dict(), f'{new_exp_dir}/last_{phase}_model.pt')

            # Calculate improvement metrics
            if training_mode == 'target_only':
                # For target-only, no base comparison possible
                metrics['best_target_val_f1_from_best_base_model'] = None
                metrics['customization_improvement_in_val_loss'] = None
                metrics['customization_improvement_in_val_f1'] = None
            else:
                if metrics['best_target_val_f1_from_best_base_model'] is not None:
                    metrics['customization_improvement_in_val_loss'] = metrics['best_target_val_f1'] - metrics['best_target_val_f1_from_best_base_model']
                    metrics['customization_improvement_in_val_f1'] = metrics['best_target_val_f1'] - metrics['best_target_val_f1_from_best_base_model']

            # Save results
            with open(f'{new_exp_dir}/metrics.json', 'w') as f:
                json.dump(metrics, f, indent=4)
            with open(f'{new_exp_dir}/losses.json', 'w') as f:
                json.dump(lossi, f, indent=4)
            with open(f'{new_exp_dir}/hyperparameters.json', 'w') as f:
                json.dump(hyperparameters, f, indent=4)
            break

    # Plotting (same as original but handle NaN values)
    plt.figure(figsize=(7.2,4.48))

    # Filter out NaN values for plotting
    base_train_loss = [x for x in lossi['base train loss'] if not np.isnan(x)]
    base_val_loss = [x for x in lossi['base val loss'] if not np.isnan(x)]
    target_train_loss = [x for x in lossi['target train loss'] if not np.isnan(x)]
    target_val_loss = [x for x in lossi['target val loss'] if not np.isnan(x)]

    if base_train_loss:
        plt.plot(range(len(base_train_loss)), base_train_loss, label='Train Loss (base)', color='b')
    if base_val_loss:
        plt.plot(range(len(base_val_loss)), base_val_loss, label='Val Loss (base)', color='b', linestyle='--')
    if target_train_loss:
        start_idx = len(lossi['target train loss']) - len(target_train_loss)
        plt.plot(range(start_idx, len(lossi['target train loss'])), target_train_loss, label='Train Loss (target)', color='g')
    if target_val_loss:
        plt.plot(range(len(target_val_loss)), target_val_loss, label='Val Loss (target)', color='g', linestyle='--')

    if metrics['transition_epoch'] is not None:
        plt.axvline(x=metrics['transition_epoch'], color='r', linestyle='--', label='Phase Transition')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.yscale('log')
    plt.title(f'Mode: {training_mode}, Phase: {phase}, Patience: {patience_counter}')
    plt.savefig(f'{new_exp_dir}/loss.jpg', bbox_inches='tight')
    plt.close()

    # F1 plot
    plt.figure(figsize=(7.2,4.48))

    base_train_f1 = [x for x in lossi['base train f1'] if not np.isnan(x)]
    base_val_f1 = [x for x in lossi['base val f1'] if not np.isnan(x)]
    target_train_f1 = [x for x in lossi['target train f1'] if not np.isnan(x)]
    target_val_f1 = [x for x in lossi['target val f1'] if not np.isnan(x)]

    if base_train_f1:
        plt.plot(range(len(base_train_f1)), base_train_f1, label='Train F1 (base)', color='b')
    if base_val_f1:
        plt.plot(range(len(base_val_f1)), base_val_f1, label='Val F1 (base)', color='b', linestyle='--')
    if target_train_f1:
        start_idx = len(lossi['target train f1']) - len(target_train_f1)
        plt.plot(range(start_idx, len(lossi['target train f1'])), target_train_f1, label='Train F1 (target)', color='r')
    if target_val_f1:
        plt.plot(range(len(target_val_f1)), target_val_f1, label='Val F1 (target)', color='r', linestyle='--')

    if metrics['transition_epoch'] is not None:
        plt.axvline(x=metrics['transition_epoch'], color='r', linestyle='--', label='Phase Transition')

    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.title(f'Mode: {training_mode}, Phase: {phase}, Patience: {patience_counter}')
    plt.savefig(f'{new_exp_dir}/f1.jpg', bbox_inches='tight')
    plt.close()

    epoch += 1
    print(f'Epoch {epoch}, Mode: {training_mode}, Phase: {phase}, Time Elapsed: {time() - start_time:.2f}s')

print(f"Training completed. Results saved to: {new_exp_dir}")