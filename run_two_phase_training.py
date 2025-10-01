#!/usr/bin/env python3
"""
Two-phase distributed training orchestrator.

Phase 1: Train unique base models in parallel
Phase 2: Run all experiment jobs (reusing cached base models)

This eliminates redundant base model training across experiments.
"""

import argparse
import json
import sys
import subprocess
from pathlib import Path

from distributed_train_helper import (
    extract_unique_base_models,
    generate_base_model_training_jobs,
    annotate_jobs_with_base_model_hash
)


def main():
    parser = argparse.ArgumentParser(
        description="Two-phase distributed training with base model caching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:

  python run_two_phase_training.py \\
      --cluster-config cluster_config.json \\
      --jobs-config jobs_config.json \\
      --script-path oct1_train.py

Phase 1: Trains all unique base models (distributed)
Phase 2: Runs all experiments (distributed, reusing base models)

All base models are cached on master and synced to workers as needed.
        """
    )

    parser.add_argument('--cluster-config', required=True,
                       help='Path to cluster configuration JSON')
    parser.add_argument('--jobs-config', required=True,
                       help='Path to experiment jobs configuration JSON')
    parser.add_argument('--script-path', default='oct1_train.py',
                       help='Path to training script on remote machines')
    parser.add_argument('--tmux-session', default='ml_training',
                       help='Name of tmux session (default: ml_training)')
    parser.add_argument('--skip-phase1', action='store_true',
                       help='Skip Phase 1 (base model training) if base models already exist')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')

    args = parser.parse_args()

    # Load jobs configuration
    print(f"Loading jobs from: {args.jobs_config}")
    with open(args.jobs_config, 'r') as f:
        experiment_jobs = json.load(f)

    print(f"Found {len(experiment_jobs)} experiment jobs\n")

    # ========================================================================
    # PHASE 1: TRAIN BASE MODELS
    # ========================================================================

    if not args.skip_phase1:
        print(f"{'='*80}")
        print(f"PHASE 1: TRAINING BASE MODELS")
        print(f"{'='*80}\n")

        # Extract unique base models
        unique_base_models = extract_unique_base_models(experiment_jobs)
        print(f"Identified {len(unique_base_models)} unique base model configurations\n")

        if len(unique_base_models) == 0:
            print("No base models needed (all jobs are target_only mode)")
            print("Skipping Phase 1\n")
        else:
            # Show base model details
            for i, (base_hash, config) in enumerate(unique_base_models.items(), 1):
                print(f"{i}. Hash: {base_hash}")
                print(f"   Participants: {config['base_participants']}")
                print(f"   Model: {config['model']}, Batch: {config['batch_size']}, LR: {config['lr']}")
                print()

            # Check if base models already exist
            existing_models = []
            missing_models = []
            for base_hash in unique_base_models.keys():
                base_model_path = Path(f"experiments/base_models/{base_hash}/best_base_model.pt")
                if base_model_path.exists():
                    existing_models.append(base_hash)
                else:
                    missing_models.append(base_hash)

            if existing_models:
                print(f"Found {len(existing_models)} existing base models:")
                for base_hash in existing_models:
                    print(f"  ✓ {base_hash}")
                print()

            if missing_models:
                print(f"Need to train {len(missing_models)} base models:")
                for base_hash in missing_models:
                    print(f"  • {base_hash}")
                print()

                # Generate base model training jobs (only for missing models)
                missing_base_configs = {h: unique_base_models[h] for h in missing_models}
                base_jobs = generate_base_model_training_jobs(missing_base_configs)

                # Save base jobs to temporary file
                base_jobs_file = 'jobs_base_models_temp.json'
                with open(base_jobs_file, 'w') as f:
                    json.dump(base_jobs, f, indent=2)

                print(f"Generated {len(base_jobs)} base model training jobs")
                print(f"Saved to: {base_jobs_file}\n")

                # Run Phase 1 distributed training
                print("Starting distributed base model training...\n")

                phase1_cmd = [
                    'python3', 'run_distributed_training.py',
                    '--cluster-config', args.cluster_config,
                    '--jobs-config', base_jobs_file,
                    '--script-path', args.script_path,
                    '--tmux-session', args.tmux_session,
                    '--log-file', 'phase1_base_models.json'
                ]

                if args.quiet:
                    phase1_cmd.append('--quiet')

                result = subprocess.run(phase1_cmd)

                if result.returncode != 0:
                    print("\nERROR: Phase 1 (base model training) failed!")
                    print("Check phase1_base_models.json for details")
                    sys.exit(1)

                print(f"\n{'='*80}")
                print(f"PHASE 1 COMPLETE: All base models trained")
                print(f"{'='*80}\n")
            else:
                print("All base models already exist. Skipping training.\n")
    else:
        print(f"{'='*80}")
        print(f"PHASE 1: SKIPPED (--skip-phase1)")
        print(f"{'='*80}\n")

    # ========================================================================
    # PHASE 2: RUN EXPERIMENTS
    # ========================================================================

    print(f"{'='*80}")
    print(f"PHASE 2: RUNNING EXPERIMENTS")
    print(f"{'='*80}\n")

    # Annotate jobs with base model hash for syncing
    annotated_jobs = annotate_jobs_with_base_model_hash(experiment_jobs)

    # Save annotated jobs
    annotated_jobs_file = 'jobs_experiments_annotated.json'
    with open(annotated_jobs_file, 'w') as f:
        json.dump(annotated_jobs, f, indent=2)

    print(f"Running {len(annotated_jobs)} experiment jobs")
    print(f"(Base models will be synced to workers as needed)\n")

    # Count jobs per base model
    jobs_per_base = {}
    for job in annotated_jobs:
        base_hash = job.get('_base_model_hash')
        if base_hash:
            jobs_per_base[base_hash] = jobs_per_base.get(base_hash, 0) + 1

    if jobs_per_base:
        print("Jobs per base model:")
        for base_hash, count in jobs_per_base.items():
            print(f"  {base_hash}: {count} jobs")
        print()

    # Run Phase 2 distributed training
    print("Starting distributed experiment execution...\n")

    phase2_cmd = [
        'python3', 'run_distributed_training.py',
        '--cluster-config', args.cluster_config,
        '--jobs-config', annotated_jobs_file,
        '--script-path', args.script_path,
        '--tmux-session', args.tmux_session,
        '--log-file', 'phase2_experiments.json'
    ]

    if args.quiet:
        phase2_cmd.append('--quiet')

    result = subprocess.run(phase2_cmd)

    if result.returncode != 0:
        print("\nWARNING: Some Phase 2 jobs may have failed")
        print("Check phase2_experiments.json for details")
        sys.exit(1)

    print(f"\n{'='*80}")
    print(f"ALL TRAINING COMPLETE!")
    print(f"{'='*80}\n")

    print("Results:")
    print(f"  Phase 1 log: phase1_base_models.json")
    print(f"  Phase 2 log: phase2_experiments.json")
    print(f"  Base models: experiments/base_models/")
    print(f"  Experiments: experiments/")


if __name__ == '__main__':
    main()
