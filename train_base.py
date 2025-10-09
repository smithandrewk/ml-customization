#!/usr/bin/env python3
"""
Train base models for shared use across multiple fine-tuning experiments.

This script trains ONLY the base model (no fine-tuning on target participant).
The trained model is saved with a hash-based identifier for reuse.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import os
import argparse
import hashlib
import json
from time import time
import random
import numpy as np

from lib.train_utils import *


def compute_base_model_hash(config):
    """
    Compute hash from base model configuration.

    Includes fold because each fold trains on different base participants
    (fold 0 excludes participant 0, fold 1 excludes participant 1, etc.)
    so they are fundamentally different models.
    """
    # Parameters that define a unique base model
    base_config = {
        'fold': config['fold'],  # Different folds = different base models
        'n_base_participants': config['n_base_participants'],
        'model': config['model'],
        'data_path': config['data_path'],
        'window_size': config['window_size'],
        'batch_size': config['batch_size'],
        'lr': config['lr'],
        'early_stopping_patience': config['early_stopping_patience'],
        'use_augmentation': config['use_augmentation'],
        'use_dilation': config.get('use_dilation', False),
        'dropout': config.get('dropout', 0.5),
        'participants': config['participants'],  # Full list affects which are base
        'seed': config.get('seed', 42),  # Different seeds = different initializations
    }

    # Add augmentation params if augmentation is enabled
    if config['use_augmentation']:
        base_config['jitter_std'] = config.get('jitter_std', 0.005)
        base_config['magnitude_range'] = config.get('magnitude_range', [0.98, 1.02])
        base_config['aug_prob'] = config.get('aug_prob', 0.3)

    # Create deterministic string representation
    config_str = json.dumps(base_config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def main():
    # Parse arguments
    argparser = argparse.ArgumentParser(description='Train base model for shared use')
    argparser = add_arguments(argparser)
    args = argparser.parse_args()

    # Check that data_path exists
    if not os.path.exists(args.data_path):
        raise ValueError(f"Data path {args.data_path} does not exist.")

    hyperparameters = vars(args)

    # Set random seeds for reproducibility
    seed = hyperparameters.get('seed', 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Extract key parameters
    fold = hyperparameters['fold']
    device = hyperparameters['device']
    batch_size = hyperparameters['batch_size']
    data_path = hyperparameters['data_path']
    experiment_prefix = hyperparameters['prefix']
    participants = hyperparameters['participants'].copy()  # Copy to avoid modifying original

    # Determine target participant and base participants
    target_participant = participants[fold]
    hyperparameters['target_participant'] = target_participant
    participants.remove(target_participant)

    # Apply n_base_participants constraint
    n_base = hyperparameters['n_base_participants']
    if n_base != 'all':
        n_base = int(n_base)
        if n_base > len(participants):
            raise ValueError(f"n_base_participants ({n_base}) cannot exceed available base participants ({len(participants)})")
        participants = participants[:n_base]

    # Validate we have at least one base participant
    if len(participants) == 0:
        raise ValueError(f"No base participants available for fold {fold}. Cannot train base model.")

    # Compute base model hash for reference (saved in metadata)
    base_model_hash = compute_base_model_hash(hyperparameters)

    print(f"\n{'='*80}")
    print(f"Base Model Training")
    print(f"{'='*80}")
    print(f"Experiment prefix: {experiment_prefix}")
    print(f"Target participant (excluded): {target_participant}")
    print(f"Base participants: {participants}")
    print(f"Number of base participants: {len(participants)}")
    print(f"Base model hash: {base_model_hash}")
    print(f"{'='*80}\n")

    # Create experiment directory using the prefix from job config
    new_exp_dir = f'experiments/{experiment_prefix}/fold{fold}_{target_participant}'

    if os.path.exists(new_exp_dir):
        print(f"Base model experiment already exists at {new_exp_dir}")
        print(f"Skipping training. Delete the directory to retrain.")
        return

    os.makedirs(new_exp_dir, exist_ok=False)
    print(f"Created experiment directory: {new_exp_dir}\n")

    # Load base participant data
    print("Loading base participant data...")
    base_train_dataset = ConcatDataset([
        TensorDataset(*torch.load(f'{data_path}/{p}_{s}.pt'))
        for p in participants
        for s in ['train', 'val']
    ])
    base_val_dataset = ConcatDataset([
        TensorDataset(*torch.load(f'{data_path}/{p}_test.pt'))
        for p in participants
    ])

    print(f'Base train dataset size: {len(base_train_dataset)}')
    base_train_dataset = random_subsample(base_train_dataset, 1)
    print(f'Base train dataset size after subsampling: {len(base_train_dataset)}')

    print(f'Base val dataset size: {len(base_val_dataset)}')
    base_val_dataset = random_subsample(base_val_dataset, 1)
    print(f'Base val dataset size after subsampling: {len(base_val_dataset)}')

    # Create data loaders
    base_trainloader = DataLoader(base_train_dataset, batch_size=batch_size, shuffle=True)
    base_valloader = DataLoader(base_val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    model_type = hyperparameters['model']
    dropout = hyperparameters.get('dropout', 0.5)
    use_dilation = hyperparameters.get('use_dilation', False)
    from lib.models import TestModel
    model = TestModel(dropout=dropout, use_dilation=use_dilation)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparameters['lr'])

    print(f'Using model: {model.__class__.__name__}')
    print(f'Total model parameters: {sum(p.numel() for p in model.parameters())}')

    # Setup augmentation if enabled
    augmenter = None
    if hyperparameters['use_augmentation']:
        from lib.train_utils import TimeSeriesAugmenter
        augmenter = TimeSeriesAugmenter({
            'jitter_noise_std': hyperparameters['jitter_std'],
            'magnitude_scale_range': hyperparameters['magnitude_range'],
            'augmentation_probability': hyperparameters['aug_prob']
        })
        print(f'Data augmentation enabled: jitter_std={hyperparameters["jitter_std"]}, '
              f'magnitude_range={hyperparameters["magnitude_range"]}, '
              f'aug_prob={hyperparameters["aug_prob"]}')

    # Training metrics
    metrics = {
        'best_val_loss': None,
        'best_val_loss_epoch': None,
        'best_val_f1': None,
        'best_val_f1_epoch': None,
        'total_epochs': 0,
    }

    lossi = {
        'train_loss': [],
        'train_f1': [],
        'val_loss': [],
        'val_f1': [],
    }

    # Training loop
    epoch = 0
    early_stopping_patience = hyperparameters['early_stopping_patience']
    early_stopping_metric = hyperparameters.get('early_stopping_metric', 'loss')

    # Initialize best metric value based on metric type
    if early_stopping_metric == 'f1':
        best_val_metric = 0.0  # F1: higher is better
    else:  # loss
        best_val_metric = float('inf')  # Loss: lower is better

    patience_counter = 0

    print(f"\nStarting base model training (max {early_stopping_patience} epochs patience)...")
    print(f"Early stopping metric: {early_stopping_metric}\n")

    while True:
        start_time = time()

        # Training step
        model.train()
        train_loss, train_f1 = optimize_model_compute_loss_and_f1(
            model, base_trainloader, optimizer, criterion,
            device=device, augmenter=augmenter
        )
        lossi['train_loss'].append(train_loss)
        lossi['train_f1'].append(train_f1)

        # Validation step
        loss, f1 = compute_loss_and_f1(model, base_valloader, criterion, device=device)
        lossi['val_loss'].append(loss)
        lossi['val_f1'].append(f1)

        # Early stopping based on selected metric
        current_metric = f1 if early_stopping_metric == 'f1' else loss

        # Check if this is the best model so far
        is_best = False
        if early_stopping_metric == 'f1':
            is_best = current_metric > best_val_metric  # F1: higher is better
        else:  # loss
            is_best = current_metric < best_val_metric  # Loss: lower is better

        if is_best:
            best_val_metric = current_metric
            torch.save(model.state_dict(), f'{new_exp_dir}/best_base_model.pt')
            patience_counter = 0
        else:
            patience_counter += 1

        # Track best metrics (for logging, independent of early stopping)
        if loss < (metrics['best_val_loss'] or float('inf')):
            metrics['best_val_loss'] = loss
            metrics['best_val_loss_epoch'] = epoch

        if f1 > (metrics['best_val_f1'] or 0):
            metrics['best_val_f1'] = f1
            metrics['best_val_f1_epoch'] = epoch

        # Check early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            torch.save(model.state_dict(), f'{new_exp_dir}/last_base_model.pt')
            break

        # Progress output and plotting
        if epoch % 5 == 0:
            print(f'Epoch {epoch:3d} | Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f} | '
                  f'Val Loss: {loss:.4f}, Val F1: {f1:.4f} | Patience: {patience_counter}/{early_stopping_patience}')
            # Plot using train.py style (but adapted for base-only training)
            plot_base_training(lossi, new_exp_dir, metrics, patience_counter)

        epoch += 1

    metrics['total_epochs'] = epoch

    # Final summary
    print(f"\n{'='*80}")
    print(f"Base Model Training Complete")
    print(f"{'='*80}")
    print(f"Experiment directory: {new_exp_dir}")
    print(f"Base model hash: {base_model_hash}")
    print(f"Best validation loss: {metrics['best_val_loss']:.4f} (epoch {metrics['best_val_loss_epoch']})")
    print(f"Best validation F1: {metrics['best_val_f1']:.4f} (epoch {metrics['best_val_f1_epoch']})")
    print(f"Total epochs trained: {metrics['total_epochs']}")
    print(f"{'='*80}\n")

    # Save metrics, losses, and hyperparameters
    metrics['base_model_hash'] = base_model_hash
    save_metrics_and_losses(metrics, lossi, hyperparameters, new_exp_dir)
    plot_base_training(lossi, new_exp_dir, metrics, patience_counter)

    print(f"Results saved to: {new_exp_dir}")


if __name__ == '__main__':
    main()
