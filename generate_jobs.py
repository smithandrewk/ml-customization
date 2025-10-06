#!/usr/bin/env python3
"""
Generate job configurations for two-phase distributed training.

Phase 1: Train unique base models
Phase 2: Fine-tune base models on different target participants/settings
"""

import json
import hashlib
from itertools import product
from collections import defaultdict
from datetime import datetime


# Grid search parameters
GRID_PARAMS = {
    'batch_size': [32],
    'lr': [3e-4],
    'early_stopping_patience': [50],
    'mode': ['full_fine_tuning','target_only','target_only_fine_tuning'],
    'target_data_pct': [0.01, 0.05, 0.125, 0.25, 0.5, 1.0],
    'n_base_participants': [6],
}

# Fixed parameters
FIXED_PARAMS = {
    'model': 'test',
    'data_path': 'data/001_60s_window',
    'participants': ['tonmoy', 'asfik', 'alsaad', 'anam', 'ejaz', 'iftakhar', 'unk1'],
    'window_size': 3000,
    'use_augmentation': True,
    'early_stopping_patience_target': 50,
    'jitter_std': 0.005,
    'magnitude_range': [0.98, 1.02],
    'aug_prob': 0.3,
}


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
        'participants': config['participants'],  # Full list affects which are base
    }

    # Add augmentation params if augmentation is enabled
    if config['use_augmentation']:
        base_config['jitter_std'] = config.get('jitter_std', 0.005)
        base_config['magnitude_range'] = config.get('magnitude_range', [0.98, 1.02])
        base_config['aug_prob'] = config.get('aug_prob', 0.3)

    # Create deterministic string representation
    config_str = json.dumps(base_config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def compute_finetune_experiment_hash(config):
    """
    Compute hash for fine-tuning experiment directory.

    Excludes fold and base model reference so all folds with same
    fine-tuning hyperparameters share one experiment directory.
    """
    finetune_config = {
        'mode': config['mode'],
        'target_data_pct': config['target_data_pct'],
        'n_base_participants': config['n_base_participants'],
        'model': config['model'],
        'data_path': config['data_path'],
        'window_size': config['window_size'],
        'batch_size': config['batch_size'],
        'lr': config['lr'],
        'early_stopping_patience_target': config['early_stopping_patience_target'],
        'use_augmentation': config['use_augmentation'],
        'participants': config['participants'],
    }

    if config['use_augmentation']:
        finetune_config['jitter_std'] = config.get('jitter_std', 0.005)
        finetune_config['magnitude_range'] = config.get('magnitude_range', [0.98, 1.02])
        finetune_config['aug_prob'] = config.get('aug_prob', 0.3)

    config_str = json.dumps(finetune_config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def generate_all_experiment_configs():
    """Generate all experiment configurations from grid search."""
    configs = []

    # Separate modes that need base participants from those that don't
    modes_needing_base = [m for m in GRID_PARAMS['mode'] if m != 'target_only']
    target_only_mode = 'target_only' in GRID_PARAMS['mode']

    # Generate configs for modes that need base participants
    if modes_needing_base:
        # Full grid search including n_base_participants
        grid_params_with_base = GRID_PARAMS.copy()
        grid_params_with_base['mode'] = modes_needing_base

        param_names = list(grid_params_with_base.keys())
        param_values = list(grid_params_with_base.values())

        for param_combo in product(*param_values):
            params = dict(zip(param_names, param_combo))

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            prefix = f"b{params['batch_size']}_t{params['target_data_pct']}_{timestamp}"

            # Run all folds
            for fold in range(len(FIXED_PARAMS['participants'])):
                config = {
                    'fold': fold,
                    'prefix': prefix,
                    **params,
                    **FIXED_PARAMS
                }
                configs.append(config)

    # Generate configs for target_only mode (no n_base_participants sweep)
    if target_only_mode:
        # Grid search WITHOUT n_base_participants (it's irrelevant for target_only)
        grid_params_target_only = {k: v for k, v in GRID_PARAMS.items() if k != 'n_base_participants'}
        grid_params_target_only['mode'] = ['target_only']

        param_names = list(grid_params_target_only.keys())
        param_values = list(grid_params_target_only.values())

        for param_combo in product(*param_values):
            params = dict(zip(param_names, param_combo))

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            prefix = f"b{params['batch_size']}_t{params['target_data_pct']}_{timestamp}"

            # Run all folds
            for fold in range(len(FIXED_PARAMS['participants'])):
                config = {
                    'fold': fold,
                    'prefix': prefix,
                    **params,
                    'n_base_participants': 'all',  # Irrelevant but needed for consistency
                    **FIXED_PARAMS
                }
                configs.append(config)

    return configs


def generate_two_phase_jobs():
    """
    Generate base training jobs and fine-tuning jobs.

    Returns:
        tuple: (base_jobs, finetune_jobs)
    """
    # Generate all experiment configs
    all_configs = generate_all_experiment_configs()

    print(f"Total experiment configurations: {len(all_configs)}")

    # Separate configs that need base models from those that don't
    # target_only mode trains from scratch, no base model needed
    configs_needing_base = [c for c in all_configs if c['mode'] != 'target_only']
    configs_no_base = [c for c in all_configs if c['mode'] == 'target_only']

    print(f"Configs needing base models: {len(configs_needing_base)}")
    print(f"Configs without base models (target_only): {len(configs_no_base)}")

    # Group configs by base model hash (only for those needing base models)
    base_model_groups = defaultdict(list)

    for config in configs_needing_base:
        base_hash = compute_base_model_hash(config)
        base_model_groups[base_hash].append(config)

    print(f"Unique base models needed: {len(base_model_groups)}")

    # Generate base training jobs (one per unique base model)
    base_jobs = []
    for base_hash, configs in base_model_groups.items():
        # Use the first config from this group to create the base training job
        # All configs in the group share the same base model parameters
        reference_config = configs[0]

        # Create base training job with only base-relevant parameters
        base_job = {
            'fold': reference_config['fold'],
            'device': 0,  # Will be set by distributed training system
            'batch_size': reference_config['batch_size'],
            'model': reference_config['model'],
            'prefix': f"base_{base_hash}",
            'lr': reference_config['lr'],
            'early_stopping_patience': reference_config['early_stopping_patience'],
            'early_stopping_patience_target': reference_config['early_stopping_patience_target'],
            'mode': 'full_fine_tuning',  # Doesn't matter for base training
            'target_data_pct': 1.0,  # Doesn't matter for base training
            'n_base_participants': reference_config['n_base_participants'],
            'data_path': reference_config['data_path'],
            'window_size': reference_config['window_size'],
            'participants': reference_config['participants'],
            'use_augmentation': reference_config['use_augmentation'],
        }

        if reference_config['use_augmentation']:
            base_job['jitter_std'] = reference_config['jitter_std']
            base_job['magnitude_range'] = reference_config['magnitude_range']
            base_job['aug_prob'] = reference_config['aug_prob']

        # Set prefix to base_{hash} so experiment directory is predictable
        base_job['prefix'] = f'base_{base_hash}'

        # Add hash for tracking
        base_job['_base_model_hash'] = base_hash
        base_job['_num_dependent_jobs'] = len(configs)

        base_jobs.append(base_job)

    # Generate fine-tuning jobs (all original configs, with base_experiment_prefix added where needed)
    finetune_jobs = []

    # Jobs that need base models
    for config in configs_needing_base:
        base_hash = compute_base_model_hash(config)
        finetune_hash = compute_finetune_experiment_hash(config)
        finetune_job = config.copy()
        # Set prefix based on fine-tuning config (all folds share same experiment dir)
        finetune_job['prefix'] = f'finetune_{finetune_hash}'
        # Store base model location for loading
        finetune_job['base_experiment_prefix'] = f'base_{base_hash}'
        finetune_job['device'] = 0  # Will be set by distributed training system
        finetune_jobs.append(finetune_job)

    # Jobs that don't need base models (target_only)
    for config in configs_no_base:
        finetune_hash = compute_finetune_experiment_hash(config)
        finetune_job = config.copy()
        # Set prefix based on fine-tuning config
        finetune_job['prefix'] = f'target_only_{finetune_hash}'
        finetune_job['base_experiment_prefix'] = None  # No base model needed
        finetune_job['device'] = 0  # Will be set by distributed training system
        finetune_jobs.append(finetune_job)

    return base_jobs, finetune_jobs


def main():
    """Generate and save job configurations."""
    base_jobs, finetune_jobs = generate_two_phase_jobs()

    # Save base training jobs
    with open('base_training_jobs.json', 'w') as f:
        json.dump(base_jobs, f, indent=2)

    # Save fine-tuning jobs
    with open('finetune_jobs.json', 'w') as f:
        json.dump(finetune_jobs, f, indent=2)

    # Print summary
    print(f"\n{'='*80}")
    print(f"Job Generation Summary")
    print(f"{'='*80}")
    print(f"Base training jobs: {len(base_jobs)}")
    print(f"Fine-tuning jobs: {len(finetune_jobs)}")
    print(f"Total jobs: {len(base_jobs) + len(finetune_jobs)}")
    print(f"\nFiles created:")
    print(f"  - base_training_jobs.json ({len(base_jobs)} jobs)")
    print(f"  - finetune_jobs.json ({len(finetune_jobs)} jobs)")
    print(f"{'='*80}\n")

    # Print example breakdown
    print("Example: Jobs sharing the same base model")
    if base_jobs:
        example_hash = base_jobs[0]['_base_model_hash']
        example_prefix = f'base_{example_hash}'
        example_jobs = [j for j in finetune_jobs if j.get('base_experiment_prefix') == example_prefix]
        print(f"Base experiment prefix: {example_prefix}")
        print(f"Number of fine-tuning jobs using this base: {len(example_jobs)}")
        print(f"Fold: {base_jobs[0]['fold']}")
        print(f"Base participants (n={base_jobs[0]['n_base_participants']})")
        print(f"\nThese {len(example_jobs)} jobs differ only in:")
        modes = set(j['mode'] for j in example_jobs)
        target_pcts = set(j['target_data_pct'] for j in example_jobs)
        print(f"  - mode: {modes}")
        print(f"  - target_data_pct: {sorted(target_pcts)}")
        print()

    print("Next steps:")
    print("1. Train base models:")
    print("   python run_distributed_training.py \\")
    print("       --cluster-config cluster_config.json \\")
    print("       --jobs-config base_training_jobs.json \\")
    print("       --script-path train_base.py")
    print()
    print("2. Fine-tune on targets:")
    print("   python run_distributed_training.py \\")
    print("       --cluster-config cluster_config.json \\")
    print("       --jobs-config finetune_jobs.json \\")
    print("       --script-path train_finetune.py")


if __name__ == '__main__':
    main()
