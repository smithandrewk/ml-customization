#!/usr/bin/env python3
"""
Helper utilities for distributed training with base model caching.

This module provides functions to:
1. Extract unique base model configurations from job lists
2. Generate base model training jobs
3. Coordinate two-phase distributed training (base models, then experiments)
"""

import json
import hashlib
from typing import List, Dict, Set, Tuple


def get_base_model_hash_from_job(job: dict) -> str:
    """
    Calculate base model hash from a job configuration.
    Only includes parameters that affect base model training.

    Args:
        job: Job configuration dictionary

    Returns:
        Hash string identifying the base model configuration
    """
    # Determine base participants
    participants = job.get('participants', ['tonmoy', 'asfik', 'ejaz'])
    fold = job['fold']
    target_participant = participants[fold]

    # Get base participants
    mode = job.get('mode', 'full_fine_tuning')
    if mode == 'target_only':
        # No base model needed
        return None

    base_participants = [p for p in participants if p != target_participant]

    # Apply n_base_participants constraint
    n_base = job.get('n_base_participants', 'all')
    if n_base != 'all':
        base_participants = base_participants[:int(n_base)]

    if not base_participants:
        return None

    # Build base model config (same logic as oct1_train.py)
    base_config = {
        'participants': sorted(base_participants),
        'model_type': job.get('model', 'test'),
        'window_size': job.get('window_size', 3000),
        'lr': job.get('lr', 3e-4),
        'batch_size': job.get('batch_size', 64),
        'data_path': job.get('data_path', 'data/001_60s_window'),
        'use_augmentation': job.get('use_augmentation', False),
        'jitter_std': job.get('jitter_std', 0.005),
        'magnitude_range': job.get('magnitude_range', [0.98, 1.02]),
        'aug_prob': job.get('aug_prob', 0.3),
        'early_stopping_patience': job.get('early_stopping_patience', 40)
    }

    # Create deterministic string representation
    config_str = json.dumps(base_config, sort_keys=True)

    # Generate hash
    hash_obj = hashlib.sha256(config_str.encode())
    return hash_obj.hexdigest()[:12]


def extract_unique_base_models(jobs: List[dict]) -> Dict[str, dict]:
    """
    Extract unique base model configurations from a list of jobs.

    Args:
        jobs: List of job configurations

    Returns:
        Dictionary mapping base model hash -> base model config
    """
    unique_base_models = {}

    for job in jobs:
        base_hash = get_base_model_hash_from_job(job)

        if base_hash is None:
            continue

        if base_hash in unique_base_models:
            continue

        # Extract base model configuration
        participants = job.get('participants', ['tonmoy', 'asfik', 'ejaz'])
        fold = job['fold']
        target_participant = participants[fold]
        base_participants = [p for p in participants if p != target_participant]

        # Apply n_base_participants constraint
        n_base = job.get('n_base_participants', 'all')
        if n_base != 'all':
            base_participants = base_participants[:int(n_base)]

        unique_base_models[base_hash] = {
            'hash': base_hash,
            'base_participants': base_participants,
            'model': job.get('model', 'test'),
            'window_size': job.get('window_size', 3000),
            'lr': job.get('lr', 3e-4),
            'batch_size': job.get('batch_size', 64),
            'data_path': job.get('data_path', 'data/001_60s_window'),
            'use_augmentation': job.get('use_augmentation', False),
            'jitter_std': job.get('jitter_std', 0.005),
            'magnitude_range': job.get('magnitude_range', [0.98, 1.02]),
            'aug_prob': job.get('aug_prob', 0.3),
            'early_stopping_patience': job.get('early_stopping_patience', 40),
            'participants': participants,  # Keep full list for compatibility
        }

    return unique_base_models


def generate_base_model_training_jobs(unique_base_models: Dict[str, dict]) -> List[dict]:
    """
    Generate minimal training jobs for base models only.

    Args:
        unique_base_models: Dictionary from extract_unique_base_models()

    Returns:
        List of job configurations for training base models
    """
    base_jobs = []

    for base_hash, config in unique_base_models.items():
        # Create a job that will train only the base model
        # We use a dummy fold that excludes one base participant
        # The base model will be trained on the remaining base participants

        participants = config['participants']
        base_participants = config['base_participants']

        # Use first base participant as "target" for the dummy job
        # This ensures the base model is trained on the correct participants
        dummy_target = base_participants[0] if base_participants else participants[0]
        dummy_fold = participants.index(dummy_target)

        job = {
            'fold': dummy_fold,
            'model': config['model'],
            'window_size': config['window_size'],
            'lr': config['lr'],
            'batch_size': config['batch_size'],
            'data_path': config['data_path'],
            'use_augmentation': config['use_augmentation'],
            'jitter_std': config['jitter_std'],
            'magnitude_range': config['magnitude_range'],
            'aug_prob': config['aug_prob'],
            'early_stopping_patience': config['early_stopping_patience'],
            'early_stopping_patience_target': config['early_stopping_patience'],
            'participants': participants,
            'n_base_participants': 'all' if len(base_participants) == len(participants) - 1 else len(base_participants),
            'mode': 'base_only',  # Special mode: train base model only, then exit
            'prefix': f'base_models_temp/{base_hash}',
            'target_data_pct': 1.0,
            'target_batch_pct': None,
            '_base_model_hash': base_hash,  # Track which base model this creates
        }

        base_jobs.append(job)

    return base_jobs


def annotate_jobs_with_base_model_hash(jobs: List[dict]) -> List[dict]:
    """
    Add base model hash to each job for tracking.

    Args:
        jobs: List of job configurations

    Returns:
        Same jobs with '_base_model_hash' field added
    """
    annotated_jobs = []

    for job in jobs:
        job_copy = job.copy()
        base_hash = get_base_model_hash_from_job(job)
        job_copy['_base_model_hash'] = base_hash
        annotated_jobs.append(job_copy)

    return annotated_jobs


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 2:
        print("Usage: python distributed_train_helper.py <jobs_config.json>")
        print("\nThis will analyze the jobs and show base model information.")
        sys.exit(1)

    jobs_file = sys.argv[1]

    # Load jobs
    with open(jobs_file, 'r') as f:
        jobs = json.load(f)

    print(f"Analyzing {len(jobs)} jobs from {jobs_file}...\n")

    # Extract unique base models
    unique_base_models = extract_unique_base_models(jobs)

    print(f"Found {len(unique_base_models)} unique base model configurations:\n")

    for i, (base_hash, config) in enumerate(unique_base_models.items(), 1):
        print(f"{i}. Hash: {base_hash}")
        print(f"   Participants: {config['base_participants']}")
        print(f"   Model: {config['model']}, Batch: {config['batch_size']}, LR: {config['lr']}")
        print(f"   Data: {config['data_path']}")
        print()

    # Generate base model training jobs
    base_jobs = generate_base_model_training_jobs(unique_base_models)

    print(f"\nGenerated {len(base_jobs)} base model training jobs")
    print(f"Save these to a file and run them first in Phase 1")

    # Count jobs that need each base model
    jobs_per_base = {}
    for job in jobs:
        base_hash = get_base_model_hash_from_job(job)
        if base_hash:
            jobs_per_base[base_hash] = jobs_per_base.get(base_hash, 0) + 1

    print(f"\nJobs per base model:")
    for base_hash, count in jobs_per_base.items():
        print(f"  {base_hash}: {count} jobs")

    print(f"\nTotal compute saved by base model reuse:")
    total_jobs = len(jobs)
    jobs_needing_base = sum(jobs_per_base.values())
    base_training_jobs = len(base_jobs)
    savings = jobs_needing_base - base_training_jobs
    print(f"  Without caching: {total_jobs} full training runs")
    print(f"  With caching: {base_training_jobs} base + {total_jobs} target = ~{base_training_jobs + total_jobs} runs")
    print(f"  Savings: ~{savings} base model training runs eliminated")
