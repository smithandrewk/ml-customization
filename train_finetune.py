#!/usr/bin/env python3
"""
Fine-tune a pre-trained base model on target participant data.

This script loads a base model trained with train_base.py and fine-tunes it
on a specific target participant's data.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import os
import argparse
import json
from time import time
import sys
import random
import numpy as np

from lib.train_utils import *


def main():
    # Parse arguments
    argparser = argparse.ArgumentParser(description='Fine-tune base model on target participant')
    argparser = add_arguments(argparser)
    argparser.add_argument('--base_experiment_prefix', type=str, required=False, default=None,
                          help='Prefix of the base experiment to load from (not needed for target_only mode)')
    args = argparser.parse_args()

    # Check that data_path exists
    if not os.path.exists(args.data_path):
        raise ValueError(f"Data path {args.data_path} does not exist.")

    hyperparameters = vars(args)

    # Set random seeds for reproducibility
    # Use seed_finetune if available (for multiple finetune runs from same base), else use seed
    seed = hyperparameters.get('seed_finetune', hyperparameters.get('seed', 42))
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
    base_experiment_prefix = hyperparameters.get('base_experiment_prefix')
    mode = hyperparameters['mode']
    target_data_pct = hyperparameters['target_data_pct']

    participants = hyperparameters['participants'].copy()
    target_participant = participants[fold]
    hyperparameters['target_participant'] = target_participant

    # Create experiment directory
    new_exp_dir = f'experiments/{experiment_prefix}/fold{fold}_{target_participant}'

    # Check if experiment already completed
    metrics_path = os.path.join(new_exp_dir, 'metrics.json')
    if os.path.exists(metrics_path):
        print(f"\n✓ Experiment already completed: {new_exp_dir}")
        print(f"  Skipping to avoid overwriting results.")
        print(f"  (Delete the directory to re-run)\n")
        sys.exit(0)

    os.makedirs(new_exp_dir, exist_ok=False)

    # Check mode and base_model_hash compatibility
    if mode == 'target_only':
        # target_only mode trains from scratch, no base model needed
        print(f"\n{'='*80}")
        print(f"Training from Scratch (target_only mode)")
        print(f"{'='*80}")
        print(f"Target participant: {target_participant}")
        print(f"Mode: {mode}")
        print(f"Target data percentage: {target_data_pct}")
        print(f"Experiment directory: {new_exp_dir}")
        print(f"{'='*80}\n")

        # Create model from scratch
        model_type = hyperparameters['model']
        dropout = hyperparameters.get('dropout', 0.5)
        from lib.models import TestModel
        model = TestModel(dropout=dropout)
        model.to(device)

        print(f"Initialized fresh model: {model.__class__.__name__}")
        print(f"Total model parameters: {sum(p.numel() for p in model.parameters())}\n")

    else:
        # Other modes require a base model
        if not base_experiment_prefix:
            raise ValueError(f"Mode '{mode}' requires --base_experiment_prefix parameter")

        print(f"\n{'='*80}")
        print(f"Fine-Tuning from Base Model")
        print(f"{'='*80}")
        print(f"Base experiment prefix: {base_experiment_prefix}")
        print(f"Target participant: {target_participant}")
        print(f"Mode: {mode}")
        print(f"Target data percentage: {target_data_pct}")
        print(f"Experiment directory: {new_exp_dir}")
        print(f"{'='*80}\n")

        # Load base model from experiments directory
        base_exp_dir = f'experiments/{base_experiment_prefix}/fold{fold}_{target_participant}'
        base_model_path = f'{base_exp_dir}/best_base_model.pt'
        metadata_path = f'{base_exp_dir}/metrics.json'

        if not os.path.exists(base_model_path):
            raise ValueError(f"Base model not found: {base_model_path}. "
                            f"Train it first with train_base.py")

        if not os.path.exists(metadata_path):
            raise ValueError(f"Base model metadata not found: {metadata_path}")

        # Load and verify metadata
        with open(metadata_path, 'r') as f:
            base_metadata = json.load(f)

        print(f"Loading base model from: {base_model_path}")
        print(f"Base model hash: {base_metadata.get('base_model_hash', 'N/A')}")
        print(f"Base model best val loss: {base_metadata.get('best_val_loss', 'N/A')}")
        print(f"Base model best val F1: {base_metadata.get('best_val_f1', 'N/A')}\n")

        # Create model and load base weights
        model_type = hyperparameters['model']
        dropout = hyperparameters.get('dropout', 0.5)
        from lib.models import TestModel
        model = TestModel(dropout=dropout)
        model.load_state_dict(torch.load(base_model_path, map_location='cpu'))
        model.to(device)

        print(f'Using model: {model.__class__.__name__}')
        print(f'Total model parameters: {sum(p.numel() for p in model.parameters())}')

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

    # Load target participant data
    print("\nLoading target participant data...")
    target_train_dataset = TensorDataset(*torch.load(f'{data_path}/{target_participant}_train.pt', map_location='cpu'))
    target_val_dataset = TensorDataset(*torch.load(f'{data_path}/{target_participant}_val.pt', map_location='cpu'))

    # Subsample target training data if specified
    if target_data_pct < 1.0:
        print(f'Target train dataset size: {len(target_train_dataset)}')
        target_train_dataset = random_subsample(target_train_dataset, target_data_pct)
        print(f'Target train dataset size after subsampling: {len(target_train_dataset)}')

    # Prepare training data based on mode
    if mode == 'full_fine_tuning':
        # Load base participant data and concatenate with target
        print("\nMode: full_fine_tuning - Training on base + target data")
        participants.remove(target_participant)

        n_base = hyperparameters['n_base_participants']
        if n_base != 'all':
            n_base = int(n_base)
            participants = participants[:n_base]

        base_train_dataset = ConcatDataset([
            TensorDataset(*torch.load(f'{data_path}/{p}_{s}.pt', map_location='cpu'))
            for p in participants
            for s in ['train', 'val']
        ])
        base_train_dataset = random_subsample(base_train_dataset, 1)

        # Concatenate base and target data
        train_dataset = ConcatDataset([base_train_dataset, target_train_dataset])
        print(f'Combined training dataset size: {len(train_dataset)}')

    elif mode in ['target_only_fine_tuning', 'target_only']:
        # Train only on target data (no base participants)
        print(f"\nMode: {mode} - Training on target data only")
        train_dataset = target_train_dataset

    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'full_fine_tuning', 'target_only_fine_tuning', or 'target_only'")

    # Create data loaders
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(target_val_dataset, batch_size=batch_size, shuffle=False)

    # Training metrics
    metrics = {
        'base_experiment_prefix': base_experiment_prefix,
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
    early_stopping_patience = hyperparameters['early_stopping_patience_target']
    early_stopping_metric = hyperparameters.get('early_stopping_metric', 'loss')

    # Initialize best metric value based on metric type
    if early_stopping_metric == 'f1':
        best_val_metric = 0.0  # F1: higher is better
    else:  # loss
        best_val_metric = float('inf')  # Loss: lower is better

    patience_counter = 0

    print(f"\nStarting fine-tuning (max {early_stopping_patience} epochs patience)...")
    print(f"Early stopping metric: {early_stopping_metric}\n")

    while True:
        start_time = time()

        # Training step
        model.train()
        train_loss, train_f1 = optimize_model_compute_loss_and_f1(
            model, trainloader, optimizer, criterion,
            device=device, augmenter=augmenter
        )
        lossi['train_loss'].append(train_loss)
        lossi['train_f1'].append(train_f1)

        # Validation step
        loss, f1 = compute_loss_and_f1(model, valloader, criterion, device=device)
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
            torch.save(model.state_dict(), f'{new_exp_dir}/best_model.pt')
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
            torch.save(model.state_dict(), f'{new_exp_dir}/last_model.pt')
            break

        # Progress output
        if epoch % 5 == 0:
            print(f'Epoch {epoch:3d} | Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f} | '
                  f'Val Loss: {loss:.4f}, Val F1: {f1:.4f} | Patience: {patience_counter}/{early_stopping_patience}')

        # Plot progress (using base_training style since we only have one phase)
        if epoch % 5 == 0:
            plot_base_training(lossi, new_exp_dir, metrics, patience_counter)

        epoch += 1

    metrics['total_epochs'] = epoch

    # Evaluate on test set
    print("\nEvaluating on test set...")
    target_testloader = DataLoader(
        TensorDataset(*torch.load(f'{data_path}/{target_participant}_test.pt', map_location='cpu')),
        batch_size=batch_size
    )

    # Load best model and evaluate
    model.load_state_dict(torch.load(f'{new_exp_dir}/best_model.pt', map_location='cpu'))
    model.to(device)
    test_loss, test_f1 = compute_loss_and_f1(model, target_testloader, criterion, device=device)
    metrics['test_loss'] = test_loss
    metrics['test_f1'] = test_f1

    # Final summary
    print(f"\n{'='*80}")
    print(f"Fine-Tuning Complete")
    print(f"{'='*80}")
    print(f"Best validation loss: {metrics['best_val_loss']:.4f} (epoch {metrics['best_val_loss_epoch']})")
    print(f"Best validation F1: {metrics['best_val_f1']:.4f} (epoch {metrics['best_val_f1_epoch']})")
    print(f"Test loss: {metrics['test_loss']:.4f}")
    print(f"Test F1: {metrics['test_f1']:.4f}")
    print(f"Total epochs trained: {metrics['total_epochs']}")
    print(f"{'='*80}\n")

    # Save final metrics and losses
    save_metrics_and_losses(metrics, lossi, hyperparameters, new_exp_dir)
    plot_base_training(lossi, new_exp_dir, metrics, patience_counter)

    print(f"Results saved to: {new_exp_dir}")


if __name__ == '__main__':
    main()
