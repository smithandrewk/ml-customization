#!/usr/bin/env python3
"""
Generate job configurations for distributed training.
Similar to run.py but outputs jobs.json instead of running directly.
"""

import json
from itertools import product

# Grid search parameters (matching run.py)
GRID_PARAMS = {
    'batch_size': [32, 64],
    'lr': [3e-4],
    'early_stopping_patience': [50],
    'mode': ['target_only','full_fine_tuning','target_only_fine_tuning'],  # 'full_fine_tuning', 'target_only', 'target_only_fine_tuning'
    'target_data_pct': [0.01,0.05,0.125,0.25, 0.5, 1.0],  # 0.05, 0.1, 0.25, 0.5, 1.0
    'n_base_participants': [1,2,3,4,5,6],  # 1, 2, 'all'
}

# Fixed parameters (matching run.py)
FIXED_PARAMS = {
    'model': 'test',
    'data_path': 'data/001_60s_window',
    'participants': ['tonmoy', 'asfik','alsaad','anam','ejaz','iftakhar','unk1'],
    'window_size': 3000,
    'use_augmentation': True,
    'early_stopping_patience_target': 50,
}

# Prefix for experiment directories
from datetime import datetime

def generate_jobs():
    """Generate all job combinations."""
    jobs = []

    # Get all combinations of grid parameters
    param_names = list(GRID_PARAMS.keys())
    param_values = list(GRID_PARAMS.values())

    for param_combo in product(*param_values):
        params = dict(zip(param_names, param_combo))

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        prefix = f"b{params['batch_size']}_t{params['target_data_pct']}_{timestamp}"

        # Run all folds
        for fold in range(len(FIXED_PARAMS['participants'])):
            job = {
                'fold': fold,
                'prefix': prefix,
                **params,
                **FIXED_PARAMS
            }
            jobs.append(job)

    return jobs

if __name__ == '__main__':
    jobs = generate_jobs()

    # Save to jobs_config.json
    with open('jobs_config.json', 'w') as f:
        json.dump(jobs, f, indent=2)

    print(f"Generated {len(jobs)} jobs")
    print(f"Saved to: jobs_config.json")
    print(f"\nRun with:")
    print(f"  python run_distributed_training.py --cluster-config cluster_config.json --jobs-config jobs_config.json")
