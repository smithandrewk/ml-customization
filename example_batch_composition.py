#!/usr/bin/env python3
"""
Example: Hyperparameter search over batch composition.

This script demonstrates how to search over different percentages of target data
in each batch during the target fine-tuning phase.
"""

import json

# Generate jobs with different batch compositions
jobs = []

# Parameters to search
batch_compositions = [0.1, 0.25, 0.5, 0.75, 0.9]  # 10%, 25%, 50%, 75%, 90% target data per batch
participants = ['tonmoy', 'asfik', 'ejaz']
batch_size = 64
prefix = 'batch_comp_search'

for target_batch_pct in batch_compositions:
    for fold in range(len(participants)):
        job = {
            'fold': fold,
            'device': 0,  # Will be set by distributed training
            'batch_size': batch_size,
            'model': 'test',
            'mode': 'full_fine_tuning',
            'target_batch_pct': target_batch_pct,
            'lr': 3e-4,
            'early_stopping_patience': 40,
            'early_stopping_patience_target': 40,
            'data_path': 'data/001_60s_window',
            'participants': participants,
            'prefix': f'{prefix}_pct{int(target_batch_pct*100)}',
            'n_base_participants': 'all',
            'target_data_pct': 1.0,
            'window_size': 3000,
        }
        jobs.append(job)

# Save jobs configuration
with open('jobs_batch_composition.json', 'w') as f:
    json.dump(jobs, f, indent=2)

print(f"Generated {len(jobs)} jobs")
print(f"  Batch compositions: {batch_compositions}")
print(f"  Folds: {len(participants)}")
print(f"  Total: {len(batch_compositions)} × {len(participants)} = {len(jobs)} jobs")
print(f"\nSaved to: jobs_batch_composition.json")
print(f"\nTo run:")
print(f"  python run_distributed_training.py \\")
print(f"    --cluster-config cluster_config.json \\")
print(f"    --jobs-config jobs_batch_composition.json \\")
print(f"    --script-path oct1_train.py")
print(f"\nNote: Each experiment will reuse the same cached base model!")
print(f"      Only the target fine-tuning phase will differ.")
