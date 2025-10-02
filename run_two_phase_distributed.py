#!/usr/bin/env python3
"""
Two-phase distributed training orchestrator.

Phase 1: Train all unique base models across cluster
Phase 2: Sync base models to all nodes, then fine-tune in parallel

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
    Sync the base_models directory to all cluster nodes.

    Args:
        cluster_config_path: Path to cluster configuration JSON
        verbose: Print detailed output

    Returns:
        True if all syncs succeeded, False otherwise
    """
    # Check if base_models directory exists
    if not os.path.exists('base_models'):
        print("✗ base_models directory not found. Nothing to sync.")
        return False

    # Count models
    model_files = [f for f in os.listdir('base_models') if f.endswith('.pt')]
    print(f"\n{'='*80}")
    print(f"Syncing Base Models to Cluster")
    print(f"{'='*80}")
    print(f"Found {len(model_files)} base models to sync")
    print(f"Directory: {os.path.abspath('base_models')}")
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

        # First, create base_models directory on remote if it doesn't exist
        ssh_cmd = ["ssh"] + ssh_opts + [host_string, "mkdir -p ~/ml-customization/base_models"]

        try:
            result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                print(f"[{host_string}] ✗ Failed to create base_models directory")
                fail_count += 1
                continue
        except Exception as e:
            print(f"[{host_string}] ✗ Error: {e}")
            fail_count += 1
            continue

        # Sync base_models directory using rsync for efficiency
        # Build rsync command with SSH options
        ssh_command = f"ssh {' '.join(ssh_opts)}"
        rsync_cmd = [
            "rsync",
            "-avz",
            "--progress",
            "-e", ssh_command,
            "base_models/",
            f"{host_string}:~/ml-customization/base_models/"
        ]

        if verbose:
            print(f"[{host_string}] Running: {' '.join(rsync_cmd)}")

        try:
            result = subprocess.run(
                rsync_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode == 0:
                print(f"[{host_string}] ✓ Base models synced successfully")
                if verbose and result.stdout:
                    print(f"[{host_string}] {result.stdout.strip()}")
                success_count += 1
            else:
                print(f"[{host_string}] ✗ Rsync failed: {result.stderr}")
                fail_count += 1

        except Exception as e:
            print(f"[{host_string}] ✗ Error during rsync: {e}")
            fail_count += 1

    # Print summary
    print(f"\n{'='*80}")
    print(f"Sync Summary")
    print(f"{'='*80}")
    print(f"Successful: {success_count}/{len(servers)}")
    print(f"Failed: {fail_count}/{len(servers)}")
    print(f"{'='*80}\n")

    return fail_count == 0


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
  3. Sync base_models/ directory to all nodes
  4. Train all fine-tuning experiments in parallel
        """
    )

    parser.add_argument('--cluster-config', required=True,
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
    print(f"\nLogs saved:")
    if not args.skip_base_training:
        print(f"  - base_training_log.json")
    print(f"  - finetune_training_log.json")
    print(f"{'='*80}\n")

    print("All experiments complete! 🎉")


if __name__ == '__main__':
    main()
