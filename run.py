"""
TODO: combine f1 and loss plots
"""

# TODO: add test set evaluation after training completes
# TODO: if target only, store metrics to target not base
"""
# TODO:
    freeze base encoder, train target only model, concat features, train a new classifier on top of frozen features, should be lower bounded by full fine tuning?
"""

# TODO: train a base model on all non-target participants, save it, then fine tune on target participant only data. There should be a way to branch this to speed up.
# TODO: when we fine tune, should we go from best base model or last base model?
# TODO: for a target participant, train a model with 1 base participant, 2 base participants, and all base participants, see how performance scales.
# TODO: add mode where during fine tuning we also train the base model but at a lower lr, like 1/10th the lr of the target model

import argparse
from lib.models import *
from datetime import datetime
import subprocess
from itertools import product

parser = argparse.ArgumentParser(description='Create participant-specific smoking detection dataset')
parser.add_argument('--config', type=str, default='configs/train_config.yaml', help='Path to config file')
parser.add_argument('--dry-run', action='store_true', help='Print commands without running them')
parser.add_argument('--device', type=int, default=0, help='GPU device index')
args = parser.parse_args()

GRID_PARAMS = {
    'batch_size': [32],
    'lr': [3e-4],
    'patience': [5],
    'mode': ['target_only'],  # 'full_fine_tuning', 'target_only', 'target_only_fine_tuning'
    'target_data_pct': [1.0],  # 0.05, 0.1, 0.25, 0.5, 1.0,
    'n_base_participants': [1],  # 1, 2, all, must be <= number of participants - 1
}

FIXED_PARAMS = {
    'device': args.device,
    'model': 'test',
    'use_augmentation': True,
    'data_path': 'data/001_60s_window',
    'participants': ['tonmoy','asfik'],
    'window_size': 3000,
}

DRY_RUN = args.dry_run

def run_experiments():
    # Check n_base_participants is valid
    n_participants = len(FIXED_PARAMS['participants'])
    for n in GRID_PARAMS['n_base_participants']:
        if n != 'all' and (not isinstance(n, int) or n < 1 or n >= n_participants):
            raise ValueError(f"Invalid n_base_participants: {n}. Must be an integer between 1 and {n_participants - 1}, or 'all'.")
        
    """Run all experiment combinations."""
    # Generate all hyperparameter combinations
    param_names = list(GRID_PARAMS.keys())
    param_values = list(GRID_PARAMS.values())

    # Iterate over all hyperparameter combinations
    for param_combo in product(*param_values):
        params = dict(zip(param_names, param_combo))

        print(f"\n{'='*80}")
        print(f"Hyperparameters: {params}")
        print(f"{'='*80}")

        batch_size = params['batch_size']
        lr = params['lr']
        patience = params['patience']
        mode = params['mode']
        target_data_pct = params['target_data_pct']

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        prefix = f"b{batch_size}_aug_patience{patience}_{mode}_pct{target_data_pct}_{timestamp}"
            
        n_base_participants = params['n_base_participants']

        for fold,participant in enumerate(FIXED_PARAMS['participants'][:1]):
            cmd = [
                'python3', 'train.py',
                '--fold', str(fold),
                '--device', str(FIXED_PARAMS['device']),
                '--batch_size', str(batch_size),
                '--model', FIXED_PARAMS['model'],
                '--prefix', prefix,
                '--early_stopping_patience', str(patience),
                '--early_stopping_patience_target', str(patience),
                '--mode', mode,
                '--lr', str(lr),
                '--target_data_pct', str(target_data_pct),
                '--n_base_participants', str(n_base_participants),
                '--participants', *FIXED_PARAMS['participants'],
                '--window_size', str(FIXED_PARAMS['window_size']),
                '--data_path', FIXED_PARAMS['data_path'],
            ]
            if FIXED_PARAMS['use_augmentation']:
                cmd.append('--use_augmentation')

            print(f"\n{'='*80}")
            print(f"Running fold {fold} for participant {participant} with prefix {prefix}")
            print(f"{'='*80}\n")
            if DRY_RUN:
                print(f"Command: {' '.join(cmd)}")
            else:
                try:
                    result = subprocess.run(cmd, check=True)
                    print(f"✓ Fold completed successfully")
                except subprocess.CalledProcessError as e:
                    print(f"✗ Fold failed with error code {e.returncode}")
                    print(f"Consider continuing or stopping the sweep.")
                    # Optionally: raise e to stop on first failure

    print(f"\n{'='*80}")
    print(f"Experiment sweep completed!")
    print(f"{'='*80}")

if __name__ == '__main__':
    run_experiments()