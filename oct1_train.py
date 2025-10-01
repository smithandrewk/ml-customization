import os
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import argparse
import json
import hashlib
from time import time
from copy import deepcopy

from lib.utils_simple import *
from lib.utils_simple import plot_loss_and_f1
from lib.train_utils import random_subsample, append_losses_and_f1, save_metrics_and_losses


class ComposedBatchDataLoader:
    """
    DataLoader that creates batches with controlled composition from two datasets.
    Each batch contains a specified percentage of samples from target dataset
    and the rest from base dataset.
    """
    def __init__(self, base_dataset, target_dataset, batch_size, target_pct, shuffle=True):
        """
        Args:
            base_dataset: Dataset for base participants
            target_dataset: Dataset for target participant
            batch_size: Total batch size
            target_pct: Percentage of batch from target dataset (0.0-1.0)
            shuffle: Whether to shuffle the datasets
        """
        self.base_dataset = base_dataset
        self.target_dataset = target_dataset
        self.batch_size = batch_size
        self.target_pct = target_pct
        self.shuffle = shuffle

        # Calculate samples per batch from each dataset
        self.target_samples_per_batch = int(batch_size * target_pct)
        self.base_samples_per_batch = batch_size - self.target_samples_per_batch

        # Create separate dataloaders
        self.base_loader = DataLoader(base_dataset, batch_size=self.base_samples_per_batch,
                                     shuffle=shuffle, drop_last=True)
        self.target_loader = DataLoader(target_dataset, batch_size=self.target_samples_per_batch,
                                       shuffle=shuffle, drop_last=True)

        # Determine number of batches (limited by smaller dataset)
        self.num_batches = min(len(self.base_loader), len(self.target_loader))

    def __iter__(self):
        base_iter = iter(self.base_loader)
        target_iter = iter(self.target_loader)

        for _ in range(self.num_batches):
            try:
                base_X, base_y = next(base_iter)
                target_X, target_y = next(target_iter)

                # Concatenate batches
                combined_X = torch.cat([base_X, target_X], dim=0)
                combined_y = torch.cat([base_y, target_y], dim=0)

                # Shuffle within the combined batch if desired
                if self.shuffle:
                    perm = torch.randperm(combined_X.size(0))
                    combined_X = combined_X[perm]
                    combined_y = combined_y[perm]

                yield combined_X, combined_y
            except StopIteration:
                break

    def __len__(self):
        return self.num_batches

# Argument parsing
argparser = argparse.ArgumentParser()
argparser.add_argument('--fold', type=int, required=False, default=0, help='Fold index for leave-one-participant-out cross-validation')
argparser.add_argument('--device', type=int, required=False, default=0, help='GPU device index')
argparser.add_argument('--batch_size', type=int, required=False, default=64, help='batch size')
argparser.add_argument('--model', type=str, default='medium', choices=[ 'test'],help='Model architecture: simple')
argparser.add_argument('--use_augmentation', action='store_true', help='Enable data augmentation')
argparser.add_argument('--jitter_std', type=float, default=0.005, help='Standard deviation for jitter noise')
argparser.add_argument('--magnitude_range', type=float, nargs=2, default=[0.98, 1.02], help='Range for magnitude scaling')
argparser.add_argument('--aug_prob', type=float, default=0.3, help='Probability of applying augmentation')
argparser.add_argument('--prefix', type=str, default='alpha', help='Experiment prefix/directory name')
argparser.add_argument('--early_stopping_patience', type=int, default=40, help='Early stopping patience for base phase')
argparser.add_argument('--early_stopping_patience_target', type=int, default=40, help='Early stopping patience for target phase')
argparser.add_argument('--mode', type=str, default='full_fine_tuning', choices=['full_fine_tuning', 'target_only', 'target_only_fine_tuning', 'base_only'], help='Mode')
argparser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
argparser.add_argument('--target_data_pct', type=float, default=1.0, help='Percentage of target training data to use (0.0-1.0)')
argparser.add_argument('--target_batch_pct', type=float, default=None, help='Percentage of each batch from target data (0.0-1.0). Only applies to full_fine_tuning mode. If None, uses standard mixed batching.')
argparser.add_argument('--participants', type=str, nargs='+', default=['tonmoy','asfik','ejaz'], help='List of participant names for cross-validation')
argparser.add_argument('--window_size', type=int, default=3000, help='Window size in samples (e.g., 3000 = 60s at 50Hz)')
argparser.add_argument('--data_path', type=str, default='data/001_60s_window', help='Path to dataset directory')
argparser.add_argument('--n_base_participants', type=str, default='all', help='Number of base participants to use (integer or "all")')
args = argparser.parse_args()

hyperparameters = {
    'fold': args.fold,
    'device': f'cuda:{args.device}',
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
    'target_batch_pct': args.target_batch_pct,
    'n_base_participants': args.n_base_participants if args.n_base_participants == 'all' else int(args.n_base_participants)
}

fold = hyperparameters['fold']
device = hyperparameters['device']
batch_size = hyperparameters['batch_size']
window_size = hyperparameters['window_size']
experiment_prefix = hyperparameters['experiment_prefix']
data_path = hyperparameters['data_path']
participants = hyperparameters['participants'].copy()
target_participant = participants[fold]
hyperparameters['target_participant'] = target_participant


def get_base_model_hash(base_participants, hyperparameters):
    """
    Generate a unique hash for base model configuration.
    Only includes parameters that affect base model training.
    """
    base_config = {
        'participants': sorted(base_participants),
        'model_type': hyperparameters['model_type'],
        'window_size': hyperparameters['window_size'],
        'lr': hyperparameters['lr'],
        'batch_size': hyperparameters['batch_size'],
        'data_path': hyperparameters['data_path'],
        'use_augmentation': hyperparameters['use_augmentation'],
        'jitter_std': hyperparameters['jitter_std'],
        'magnitude_range': hyperparameters['magnitude_range'],
        'aug_prob': hyperparameters['aug_prob'],
        'early_stopping_patience': hyperparameters['early_stopping_patience']
    }

    # Create deterministic string representation
    config_str = json.dumps(base_config, sort_keys=True)

    # Generate hash
    hash_obj = hashlib.sha256(config_str.encode())
    return hash_obj.hexdigest()[:12]  # Use first 12 chars for readability


def train_base_model(base_participants, base_train_dataset, base_val_dataset, hyperparameters):
    """
    Train a model on base participants only.

    Returns:
        base_model_dir: Path to saved base model directory
        base_lossi: Training history for base phase
        base_metrics: Metrics for base phase
    """
    print(f"\n{'='*80}")
    print(f"TRAINING BASE MODEL")
    print(f"Base participants: {base_participants}")
    print(f"{'='*80}\n")

    # Create base model directory
    base_hash = get_base_model_hash(base_participants, hyperparameters)
    base_model_dir = f'experiments/base_models/{base_hash}'
    os.makedirs(base_model_dir, exist_ok=True)

    # Save base config
    base_config = {
        'participants': base_participants,
        'hash': base_hash,
        'hyperparameters': {k: v for k, v in hyperparameters.items()
                          if k not in ['fold', 'target_participant', 'experiment_prefix', 'target_data_pct', 'mode']}
    }
    with open(f'{base_model_dir}/base_config.json', 'w') as f:
        json.dump(base_config, f, indent=4)

    # Initialize model
    from lib.models import TestModel
    model = TestModel()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparameters['lr'])

    # Setup augmentation
    augmenter = None
    if hyperparameters['use_augmentation']:
        from lib.utils import TimeSeriesAugmenter
        augmenter = TimeSeriesAugmenter({
            'jitter_noise_std': hyperparameters['jitter_std'],
            'magnitude_scale_range': hyperparameters['magnitude_range'],
            'augmentation_probability': hyperparameters['aug_prob']
        })

    model.to(device)
    print(f'Total model parameters: {sum(p.numel() for p in model.parameters())}')

    # Subsample datasets
    print(f'Base train dataset size: {len(base_train_dataset)}')
    base_train_dataset = random_subsample(base_train_dataset, 1)
    print(f'Base train dataset size after subsampling: {len(base_train_dataset)}')

    print(f'Base val dataset size: {len(base_val_dataset)}')
    base_val_dataset = random_subsample(base_val_dataset, 0.5)
    print(f'Base val dataset size after subsampling: {len(base_val_dataset)}')

    base_trainloader = DataLoader(base_train_dataset, batch_size=batch_size, shuffle=True)
    base_valloader = DataLoader(base_val_dataset, batch_size=batch_size, shuffle=False)

    # Training loop
    base_lossi = {
        'base train loss': [],
        'base train f1': [],
        'base val loss': [],
        'base val f1': [],
    }

    base_metrics = {
        'best_base_val_loss': None,
        'best_base_val_loss_epoch': None,
        'best_base_val_f1': None,
        'best_base_val_f1_epoch': None,
    }

    epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0

    while True:
        if epoch >= 500:
            print("Maximum epochs reached.")
            torch.save(model.state_dict(), f'{base_model_dir}/last_base_model.pt')
            break

        start_time = time()
        model.train()
        train_loss, train_f1 = optimize_model_compute_loss_and_f1(
            model, base_trainloader, optimizer, criterion, device=device, augmenter=augmenter
        )

        base_lossi['base train loss'].append(train_loss)
        base_lossi['base train f1'].append(train_f1)

        val_loss, val_f1 = compute_loss_and_f1(model, base_valloader, criterion, device=device)
        base_lossi['base val loss'].append(val_loss)
        base_lossi['base val f1'].append(val_f1)

        # Early stopping on val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            base_metrics['best_base_val_loss'] = best_val_loss
            base_metrics['best_base_val_loss_epoch'] = epoch
            torch.save(model.state_dict(), f'{base_model_dir}/best_base_model.pt')
            patience_counter = 0
        else:
            patience_counter += 1

        # Track best F1
        if val_f1 > (base_metrics['best_base_val_f1'] or 0):
            base_metrics['best_base_val_f1'] = val_f1
            base_metrics['best_base_val_f1_epoch'] = epoch

        if patience_counter >= hyperparameters['early_stopping_patience']:
            print(f"Early stopping triggered at epoch {epoch}")
            torch.save(model.state_dict(), f'{base_model_dir}/last_base_model.pt')
            break

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Phase: base, Time: {time() - start_time:.2f}s, '
                  f'Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}, '
                  f'Patience: {patience_counter}')

        epoch += 1

    # Save base model metrics and losses
    with open(f'{base_model_dir}/base_metrics.json', 'w') as f:
        json.dump(base_metrics, f, indent=4)
    with open(f'{base_model_dir}/base_losses.json', 'w') as f:
        json.dump(base_lossi, f, indent=4)

    print(f"\nBase model saved to: {base_model_dir}")
    print(f"Best base val loss: {base_metrics['best_base_val_loss']:.4f} at epoch {base_metrics['best_base_val_loss_epoch']}")
    print(f"Best base val F1: {base_metrics['best_base_val_f1']:.4f} at epoch {base_metrics['best_base_val_f1_epoch']}\n")

    return base_model_dir, base_lossi, base_metrics


def load_or_train_base_model(base_participants, base_train_dataset, base_val_dataset, hyperparameters):
    """
    Check if base model exists. If yes, load it. If no, train it.

    Returns:
        base_model_dir: Path to base model directory
        base_lossi: Training history for base phase
        base_metrics: Metrics for base phase
    """
    base_hash = get_base_model_hash(base_participants, hyperparameters)
    base_model_dir = f'experiments/base_models/{base_hash}'

    if os.path.exists(f'{base_model_dir}/best_base_model.pt'):
        print(f"\n{'='*80}")
        print(f"LOADING CACHED BASE MODEL")
        print(f"Base model hash: {base_hash}")
        print(f"Base model directory: {base_model_dir}")
        print(f"{'='*80}\n")

        # Load metrics and losses
        with open(f'{base_model_dir}/base_metrics.json', 'r') as f:
            base_metrics = json.load(f)
        with open(f'{base_model_dir}/base_losses.json', 'r') as f:
            base_lossi = json.load(f)

        print(f"Loaded base model with:")
        print(f"  Best val loss: {base_metrics['best_base_val_loss']:.4f} at epoch {base_metrics['best_base_val_loss_epoch']}")
        print(f"  Best val F1: {base_metrics['best_base_val_f1']:.4f} at epoch {base_metrics['best_base_val_f1_epoch']}\n")

        return base_model_dir, base_lossi, base_metrics
    else:
        print(f"No cached base model found for hash {base_hash}. Training new base model...")
        return train_base_model(base_participants, base_train_dataset, base_val_dataset, hyperparameters)


def train_target_model(base_model_dir, base_lossi, base_metrics,
                       target_train_dataset, target_val_dataset,
                       base_train_dataset, hyperparameters, exp_dir):
    """
    Train/fine-tune on target participant data.

    Args:
        base_model_dir: Path to base model (None for target_only mode)
        base_lossi: Base training history (empty dict for target_only mode)
        base_metrics: Base metrics (empty dict for target_only mode)
        target_train_dataset: Target participant training data
        target_val_dataset: Target participant validation data
        base_train_dataset: Base participants training data (for full_fine_tuning mode)
        hyperparameters: Training hyperparameters
        exp_dir: Experiment directory for saving results

    Returns:
        metrics: Combined metrics
        lossi: Combined training history
    """
    print(f"\n{'='*80}")
    print(f"TRAINING TARGET MODEL")
    print(f"Mode: {hyperparameters['mode']}")
    print(f"Target participant: {hyperparameters['target_participant']}")
    print(f"{'='*80}\n")

    # Initialize model
    from lib.models import TestModel
    model = TestModel()

    # Load base model if available
    transition_epoch = None
    if base_model_dir is not None:
        print(f"Loading base model from: {base_model_dir}")
        model.load_state_dict(torch.load(f'{base_model_dir}/best_base_model.pt'))
        transition_epoch = len(base_lossi['base train loss'])
        print(f"Transition epoch: {transition_epoch}\n")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparameters['lr'])

    # Setup augmentation
    augmenter = None
    if hyperparameters['use_augmentation']:
        from lib.utils import TimeSeriesAugmenter
        augmenter = TimeSeriesAugmenter({
            'jitter_noise_std': hyperparameters['jitter_std'],
            'magnitude_scale_range': hyperparameters['magnitude_range'],
            'augmentation_probability': hyperparameters['aug_prob']
        })

    model.to(device)

    # Subsample target datasets
    target_data_pct = hyperparameters['target_data_pct']
    if target_data_pct < 1.0:
        print(f'Target train dataset size: {len(target_train_dataset)}')
        target_train_dataset = random_subsample(target_train_dataset, target_data_pct)
        print(f'Target train dataset size after subsampling: {len(target_train_dataset)}')

    print(f'Target val dataset size: {len(target_val_dataset)}')
    target_val_dataset = random_subsample(target_val_dataset, 0.1)
    print(f'Target val dataset size after subsampling: {len(target_val_dataset)}')

    # Determine training dataloader based on mode and batch composition settings
    target_valloader = DataLoader(target_val_dataset, batch_size=batch_size, shuffle=False)

    if hyperparameters['mode'] == 'target_only':
        train_dataset = target_train_dataset
        trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    elif hyperparameters['mode'] == 'target_only_fine_tuning':
        train_dataset = target_train_dataset
        trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    elif hyperparameters['mode'] == 'full_fine_tuning':
        # Check if using composed batching
        target_batch_pct = hyperparameters.get('target_batch_pct')
        if target_batch_pct is not None:
            print(f"Using composed batch loading: {target_batch_pct*100:.1f}% target data per batch")
            print(f"  Batch size: {batch_size}")
            print(f"  Target samples per batch: {int(batch_size * target_batch_pct)}")
            print(f"  Base samples per batch: {batch_size - int(batch_size * target_batch_pct)}")
            trainloader = ComposedBatchDataLoader(
                base_train_dataset, target_train_dataset,
                batch_size=batch_size, target_pct=target_batch_pct, shuffle=True
            )
        else:
            print("Using standard concatenated dataset (mixed batching)")
            train_dataset = ConcatDataset([base_train_dataset, target_train_dataset])
            trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize loss tracking
    lossi = deepcopy(base_lossi) if base_lossi else {
        'base train loss': [],
        'base train f1': [],
        'base val loss': [],
        'base val f1': [],
        'target train loss': [],
        'target train f1': [],
        'target val loss': [],
        'target val f1': [],
    }

    # Pad target losses with None for base phase epochs
    if base_lossi:
        lossi['target train loss'] = [None] * transition_epoch
        lossi['target train f1'] = [None] * transition_epoch
        lossi['target val loss'] = [None] * transition_epoch
        lossi['target val f1'] = [None] * transition_epoch

    metrics = deepcopy(base_metrics) if base_metrics else {}
    metrics.update({
        'transition_epoch': transition_epoch,
        'best_target_val_loss': None,
        'best_target_val_loss_epoch': None,
        'best_target_val_f1': None,
        'best_target_val_f1_epoch': None,
    })

    # Training loop
    epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0

    while True:
        if epoch >= 500:
            print("Maximum epochs reached.")
            torch.save(model.state_dict(), f'{exp_dir}/last_target_model.pt')
            break

        start_time = time()
        model.train()
        train_loss, train_f1 = optimize_model_compute_loss_and_f1(
            model, trainloader, optimizer, criterion, device=device, augmenter=augmenter
        )

        # Append training metrics
        if hyperparameters['mode'] == 'target_only':
            lossi['target train loss'].append(train_loss)
            lossi['target train f1'].append(train_f1)
        elif hyperparameters['mode'] == 'target_only_fine_tuning':
            lossi['target train loss'].append(train_loss)
            lossi['target train f1'].append(train_f1)
            lossi['base train loss'].append(None)
            lossi['base train f1'].append(None)
        elif hyperparameters['mode'] == 'full_fine_tuning':
            lossi['base train loss'].append(train_loss)
            lossi['base train f1'].append(train_f1)

        # Validation
        val_loss, val_f1 = compute_loss_and_f1(model, target_valloader, criterion, device=device)
        lossi['target val loss'].append(val_loss)
        lossi['target val f1'].append(val_f1)

        # Pad base val if needed
        if hyperparameters['mode'] in ['target_only', 'target_only_fine_tuning']:
            lossi['base val loss'].append(None)
            lossi['base val f1'].append(None)

        # Early stopping on target val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            metrics['best_target_val_loss'] = best_val_loss
            metrics['best_target_val_loss_epoch'] = epoch + (transition_epoch or 0)
            torch.save(model.state_dict(), f'{exp_dir}/best_target_model.pt')
            patience_counter = 0
        else:
            patience_counter += 1

        # Track best F1
        if val_f1 > (metrics['best_target_val_f1'] or 0):
            metrics['best_target_val_f1'] = val_f1
            metrics['best_target_val_f1_epoch'] = epoch + (transition_epoch or 0)

        if patience_counter >= hyperparameters['early_stopping_patience_target']:
            print(f"Early stopping triggered at epoch {epoch}")
            torch.save(model.state_dict(), f'{exp_dir}/last_target_model.pt')
            break

        if epoch % 10 == 0:
            plot_loss_and_f1(lossi, exp_dir, metrics, patience_counter)

        print(f'Epoch {epoch}, Phase: target, Time: {time() - start_time:.2f}s, '
              f'Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}, '
              f'Patience: {patience_counter}')

        epoch += 1

    return metrics, lossi


# ============================================================================
# MAIN TRAINING FLOW
# ============================================================================

# Create experiment directory
new_exp_dir = f'experiments/{experiment_prefix}/fold{fold}_{target_participant}'
os.makedirs(new_exp_dir, exist_ok=False)

# Determine base participants
if hyperparameters['mode'] == 'target_only':
    print(f"Target-only mode: training only on {target_participant} data.")
    base_participants = []
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

    base_participants = participants

# Load datasets
if base_participants:
    base_train_dataset = ConcatDataset([TensorDataset(*torch.load(f'{data_path}/{p}_{s}.pt'))
                                       for p in base_participants for s in ['train', 'val']])
    base_val_dataset = ConcatDataset([TensorDataset(*torch.load(f'{data_path}/{p}_test.pt'))
                                     for p in base_participants])
else:
    base_train_dataset = None
    base_val_dataset = None

target_train_dataset = TensorDataset(*torch.load(f'{data_path}/{target_participant}_train.pt'))
target_val_dataset = TensorDataset(*torch.load(f'{data_path}/{target_participant}_val.pt'))

# Handle base_only mode (for distributed base model training)
if hyperparameters['mode'] == 'base_only':
    print(f"\n{'='*80}")
    print(f"BASE-ONLY MODE: Training base model and exiting")
    print(f"{'='*80}\n")

    if not base_participants:
        print("ERROR: base_only mode requires base participants")
        exit(1)

    # Train base model
    base_model_dir, base_lossi, base_metrics = train_base_model(
        base_participants, base_train_dataset, base_val_dataset, hyperparameters
    )

    print(f"\nBase model training complete!")
    print(f"Saved to: {base_model_dir}")
    print(f"Best base val loss: {base_metrics['best_base_val_loss']:.4f}")
    print(f"Best base val F1: {base_metrics['best_base_val_f1']:.4f}")

    # Exit - no target training needed
    exit(0)

# Train or load base model
if hyperparameters['mode'] in ['full_fine_tuning', 'target_only_fine_tuning']:
    base_model_dir, base_lossi, base_metrics = load_or_train_base_model(
        base_participants, base_train_dataset, base_val_dataset, hyperparameters
    )
else:
    base_model_dir = None
    base_lossi = {}
    base_metrics = {}

# Train target model
metrics, lossi = train_target_model(
    base_model_dir, base_lossi, base_metrics,
    target_train_dataset, target_val_dataset,
    base_train_dataset, hyperparameters, new_exp_dir
)

# ============================================================================
# FINAL EVALUATION
# ============================================================================

print(f"\n{'='*80}")
print(f"FINAL EVALUATION")
print(f"{'='*80}\n")

target_testloader = DataLoader(
    TensorDataset(*torch.load(f'{data_path}/{target_participant}_test.pt')),
    batch_size=batch_size
)

criterion = nn.BCEWithLogitsLoss()

# Evaluate best base model on target test set (only for full_fine_tuning)
if hyperparameters['mode'] == 'full_fine_tuning' and base_model_dir is not None:
    from lib.models import TestModel
    model = TestModel()
    model.load_state_dict(torch.load(f'{base_model_dir}/best_base_model.pt'))
    model.to(device)
    test_loss, test_f1 = compute_loss_and_f1(model, target_testloader, criterion, device=device)
    metrics['best_base_model_target_test_loss'] = test_loss
    metrics['best_base_model_target_test_f1'] = test_f1
    print(f"Best base model on target test: Loss={test_loss:.4f}, F1={test_f1:.4f}")

# Evaluate best target model on target test set
from lib.models import TestModel
model = TestModel()
model.load_state_dict(torch.load(f'{new_exp_dir}/best_target_model.pt'))
model.to(device)
test_loss, test_f1 = compute_loss_and_f1(model, target_testloader, criterion, device=device)
metrics['best_target_model_target_test_loss'] = test_loss
metrics['best_target_model_target_test_f1'] = test_f1
print(f"Best target model on target test: Loss={test_loss:.4f}, F1={test_f1:.4f}")

# Add validation metrics
if hyperparameters['mode'] == 'full_fine_tuning' and base_metrics:
    metrics['best_base_model_target_val_loss'] = lossi['target val loss'][base_metrics['best_base_val_loss_epoch']]
    metrics['best_base_model_target_val_f1'] = lossi['target val f1'][base_metrics['best_base_val_loss_epoch']]

if metrics['best_target_val_loss_epoch'] is not None:
    metrics['best_target_model_target_val_loss'] = lossi['target val loss'][metrics['best_target_val_loss_epoch']]
    metrics['best_target_model_target_val_f1'] = lossi['target val f1'][metrics['best_target_val_loss_epoch']]

# Save final results
plot_loss_and_f1(lossi, new_exp_dir, metrics, 0)
save_metrics_and_losses(metrics, lossi, hyperparameters, new_exp_dir)

print(f"\n{'='*80}")
print(f"TRAINING COMPLETE")
print(f"Results saved to: {new_exp_dir}")
print(f"{'='*80}\n")
