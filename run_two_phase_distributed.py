#!/usr/bin/env python3
"""
Two-phase distributed training orchestrator.

Phase 1: Train all unique base models across cluster (saved to experiments/base_{hash}/)
Phase 2: Sync base model experiments to all nodes, then fine-tune in parallel

This wrapper coordinates the entire two-phase workflow automatically.
"""

import argparse
import subprocess
import json
import sys
import os
from typing import List, Dict
from datetime import datetime


def run_command(cmd: List[str], description: str, timeout: int = None) -> tuple:
    """
    Run a command and return (returncode, stdout, stderr).

    Args:
        cmd: Command and arguments as list
        description: Description for logging
        timeout: Optional timeout in seconds

    Returns:
        Tuple of (returncode, stdout, stderr)
    """
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Running: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout
        )

        # Print output in real-time fashion (though it's buffered here)
        if result.stdout:
            print(result.stdout)

        if result.returncode != 0:
            print(f"\n⚠️  Command failed with exit code {result.returncode}")
        else:
            print(f"\n✓ {description} completed successfully")

        return result.returncode, result.stdout, result.stderr

    except subprocess.TimeoutExpired as e:
        print(f"\n✗ Command timed out after {timeout} seconds")
        return -1, "", "Timeout"
    except Exception as e:
        print(f"\n✗ Error running command: {e}")
        return -1, "", str(e)


def sync_base_models_to_cluster(cluster_config_path: str, verbose: bool = True) -> bool:
    """
    Sync base model experiment directories to all cluster nodes.

    Args:
        cluster_config_path: Path to cluster configuration JSON
        verbose: Print detailed output

    Returns:
        True if all syncs succeeded, False otherwise
    """
    # Check if experiments directory exists and find base model directories
    if not os.path.exists('experiments'):
        print("✗ experiments directory not found. Nothing to sync.")
        return False

    # Find all base model experiment directories (start with "base_")
    base_exp_dirs = [d for d in os.listdir('experiments') if d.startswith('base_') and os.path.isdir(f'experiments/{d}')]

    if not base_exp_dirs:
        print("✗ No base model experiments found in experiments/ directory. Nothing to sync.")
        return False

    print(f"\n{'='*80}")
    print(f"Syncing Base Model Experiments to Cluster")
    print(f"{'='*80}")
    print(f"Found {len(base_exp_dirs)} base model experiments to sync:")
    for exp_dir in base_exp_dirs:
        print(f"  - {exp_dir}")
    print(f"{'='*80}\n")

    # Load cluster configuration
    with open(cluster_config_path) as f:
        cluster_config = json.load(f)

    servers = cluster_config['servers']
    success_count = 0
    fail_count = 0

    for server in servers:
        host_string = f"{server.get('user')}@{server['host']}" if server.get('user') else server['host']
        print(f"\n--- Syncing to {host_string} ---")

        # Build SSH/SCP commands
        ssh_opts = ["-o", "StrictHostKeyChecking=no"]

        if server.get('port', 22) != 22:
            ssh_opts.extend(["-p", str(server['port'])])

        if server.get('ssh_key'):
            ssh_opts.extend(["-i", server['ssh_key']])

        # First, create experiments directory on remote if it doesn't exist
        ssh_cmd = ["ssh"] + ssh_opts + [host_string, "mkdir -p ~/ml-customization/experiments"]

        try:
            result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                print(f"[{host_string}] ✗ Failed to create experiments directory")
                fail_count += 1
                continue
        except Exception as e:
            print(f"[{host_string}] ✗ Error: {e}")
            fail_count += 1
            continue

        # Sync each base model experiment directory using rsync for efficiency
        server_success = True
        for exp_dir in base_exp_dirs:
            # Build rsync command with SSH options
            ssh_command = f"ssh {' '.join(ssh_opts)}"
            rsync_cmd = [
                "rsync",
                "-avz",
                "--progress",
                "-e", ssh_command,
                f"experiments/{exp_dir}/",
                f"{host_string}:~/ml-customization/experiments/{exp_dir}/"
            ]

            if verbose:
                print(f"[{host_string}] Syncing {exp_dir}...")

            try:
                result = subprocess.run(
                    rsync_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=300  # 5 minute timeout
                )

                if result.returncode == 0:
                    print(f"[{host_string}] ✓ {exp_dir} synced successfully")
                else:
                    print(f"[{host_string}] ✗ Rsync failed for {exp_dir}: {result.stderr}")
                    server_success = False

            except Exception as e:
                print(f"[{host_string}] ✗ Error during rsync for {exp_dir}: {e}")
                server_success = False

        if server_success:
            success_count += 1
        else:
            fail_count += 1

    # Print summary
    print(f"\n{'='*80}")
    print(f"Sync Summary")
    print(f"{'='*80}")
    print(f"Successful: {success_count}/{len(servers)}")
    print(f"Failed: {fail_count}/{len(servers)}")
    print(f"{'='*80}\n")

    return fail_count == 0


def calculate_time_saved(base_log_path: str, finetune_jobs_path: str) -> str:
    """
    Calculate time saved by sharing base models.

    Returns formatted string showing time savings, or None if calculation fails.
    """
    try:
        # Load base training log
        with open(base_log_path, 'r') as f:
            base_log = json.load(f)

        # Load fine-tuning jobs to count reuse
        with open(finetune_jobs_path, 'r') as f:
            finetune_jobs = json.load(f)

        # Map base model prefix to training duration
        base_model_durations = {}
        for result in base_log['results']:
            if result['success']:
                prefix = result['config']['prefix']
                duration = result['duration_seconds']
                base_model_durations[prefix] = duration

        # Count how many fine-tuning jobs use each base model (exclude target_only)
        base_model_usage = {}
        for job in finetune_jobs:
            base_prefix = job.get('base_experiment_prefix')
            if base_prefix:  # Skip target_only jobs (base_experiment_prefix is None)
                if base_prefix not in base_model_usage:
                    base_model_usage[base_prefix] = 0
                base_model_usage[base_prefix] += 1

        # Calculate time saved for each base model
        total_time_saved = 0
        savings_breakdown = []

        for base_prefix, num_reuses in sorted(base_model_usage.items()):
            if base_prefix in base_model_durations:
                duration = base_model_durations[base_prefix]
                # Time saved = (num_reuses - 1) × base_model_duration
                # We subtract 1 because we still train the base model once
                time_saved = (num_reuses - 1) * duration
                total_time_saved += time_saved

                savings_breakdown.append(
                    f"  {base_prefix}:\n"
                    f"    Base model training time: {duration:.1f}s ({duration/60:.1f}min)\n"
                    f"    Used by {num_reuses} fine-tuning jobs\n"
                    f"    Time saved: {time_saved:.1f}s ({time_saved/60:.1f}min) "
                    f"[= {num_reuses-1} × {duration:.1f}s]"
                )

        # Format output
        if not savings_breakdown:
            return None

        output = "\n".join(savings_breakdown)
        output += f"\n\n  TOTAL TIME SAVED: {total_time_saved:.1f}s ({total_time_saved/60:.1f}min)"
        output += f"\n\n  💡 Without shared base models, you would have spent an additional"
        output += f"\n     {total_time_saved/60:.1f} minutes training the same base models repeatedly!"

        return output

    except Exception as e:
        print(f"Warning: Could not calculate time saved: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Two-phase distributed training orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:

  # Run complete two-phase workflow
  python run_two_phase_distributed.py \\
      --cluster-config cluster_config.json \\
      --base-jobs base_training_jobs.json \\
      --finetune-jobs finetune_jobs.json

  # Skip base training (use existing base models)
  python run_two_phase_distributed.py \\
      --cluster-config cluster_config.json \\
      --finetune-jobs finetune_jobs.json \\
      --skip-base-training

Workflow:
  1. Train all unique base models across cluster
  2. Wait for completion
  3. Sync base model experiments to all nodes
  4. Train all fine-tuning experiments in parallel
        """
    )

    parser.add_argument('--cluster-config', required=False, default='cluster_config.json',
                       help='Path to cluster configuration JSON')
    parser.add_argument('--base-jobs',
                       help='Path to base training jobs JSON (default: base_training_jobs.json)')
    parser.add_argument('--finetune-jobs',
                       help='Path to fine-tuning jobs JSON (default: finetune_jobs.json)')
    parser.add_argument('--skip-base-training', action='store_true',
                       help='Skip base training phase (use existing base models)')
    parser.add_argument('--skip-sync', action='store_true',
                       help='Skip syncing base models (assume already synced)')
    parser.add_argument('--tmux-session', default='ml_training',
                       help='Name of tmux session (default: ml_training)')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')

    args = parser.parse_args()

    # Set default job file paths
    base_jobs_path = args.base_jobs or 'base_training_jobs.json'
    finetune_jobs_path = args.finetune_jobs or 'finetune_jobs.json'

    start_time = datetime.now()

    print(f"\n{'='*80}")
    print(f"Two-Phase Distributed Training")
    print(f"{'='*80}")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Cluster config: {args.cluster_config}")
    print(f"Base jobs: {base_jobs_path}")
    print(f"Fine-tune jobs: {finetune_jobs_path}")
    print(f"Skip base training: {args.skip_base_training}")
    print(f"Skip sync: {args.skip_sync}")
    print(f"{'='*80}\n")

    # ========================================================================
    # PHASE 1: Train Base Models
    # ========================================================================

    if not args.skip_base_training:
        if not os.path.exists(base_jobs_path):
            print(f"✗ Base jobs file not found: {base_jobs_path}")
            sys.exit(1)

        print("\n" + "="*80)
        print("PHASE 1: Training Base Models")
        print("="*80 + "\n")

        base_cmd = [
            "python3", "run_distributed_training.py",
            "--cluster-config", args.cluster_config,
            "--jobs-config", base_jobs_path,
            "--script-path", "train_base.py",
            "--tmux-session", args.tmux_session,
            "--log-file", "base_training_log.json"
        ]

        if args.quiet:
            base_cmd.append("--quiet")

        returncode, stdout, stderr = run_command(
            base_cmd,
            "Phase 1: Base Model Training",
            timeout=7200  # 2 hour timeout
        )

        if returncode != 0:
            print("\n✗ Base model training failed. Aborting.")
            sys.exit(1)

        print("\n✓ All base models trained successfully!")
    else:
        print("\n⏭️  Skipping base model training (using existing base models)")

    # ========================================================================
    # SYNC: Distribute Base Models to All Nodes
    # ========================================================================

    if not args.skip_sync:
        print("\n" + "="*80)
        print("SYNC: Distributing Base Models to Cluster Nodes")
        print("="*80 + "\n")

        sync_success = sync_base_models_to_cluster(args.cluster_config, verbose=not args.quiet)

        if not sync_success:
            print("\n⚠️  Base model sync had failures. Continuing anyway...")
            print("Fine-tuning may fail if base models are not available on worker nodes.")
            response = input("Continue with fine-tuning? [y/N]: ")
            if response.lower() != 'y':
                print("Aborting.")
                sys.exit(1)
        else:
            print("\n✓ All base models synced successfully!")
    else:
        print("\n⏭️  Skipping base model sync (assuming already synced)")

    # ========================================================================
    # PHASE 2: Fine-Tuning
    # ========================================================================

    if not os.path.exists(finetune_jobs_path):
        print(f"✗ Fine-tune jobs file not found: {finetune_jobs_path}")
        sys.exit(1)

    print("\n" + "="*80)
    print("PHASE 2: Fine-Tuning on Target Participants")
    print("="*80 + "\n")

    finetune_cmd = [
        "python3", "run_distributed_training.py",
        "--cluster-config", args.cluster_config,
        "--jobs-config", finetune_jobs_path,
        "--script-path", "train_finetune.py",
        "--tmux-session", args.tmux_session,
        "--log-file", "finetune_training_log.json"
    ]

    if args.quiet:
        finetune_cmd.append("--quiet")

    returncode, stdout, stderr = run_command(
        finetune_cmd,
        "Phase 2: Fine-Tuning",
        timeout=14400  # 4 hour timeout
    )

    if returncode != 0:
        print("\n✗ Fine-tuning failed.")
        sys.exit(1)

    print("\n✓ All fine-tuning jobs completed successfully!")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================

    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()

    print(f"\n{'='*80}")
    print(f"Two-Phase Training Complete!")
    print(f"{'='*80}")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total wall time: {total_duration:.1f}s ({total_duration/60:.1f}min)")

    # Calculate and display time saved by two-phase approach
    if not args.skip_base_training and os.path.exists('base_training_log.json'):
        time_saved = calculate_time_saved('base_training_log.json', finetune_jobs_path)
        if time_saved is not None:
            print(f"\n{'='*80}")
            print(f"Time Saved by Shared Base Models")
            print(f"{'='*80}")
            print(time_saved)

    print(f"\nLogs saved:")
    if not args.skip_base_training:
        print(f"  - base_training_log.json")
    print(f"  - finetune_training_log.json")
    print(f"{'='*80}\n")

    print("All experiments complete! 🎉")


if __name__ == '__main__':
    main()
