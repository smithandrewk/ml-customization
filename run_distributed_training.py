#!/usr/bin/env python3
"""
Script to distribute training jobs across multiple servers and GPUs.
Runs jobs in parallel, maximizing GPU utilization across the cluster.
"""

import argparse
import subprocess
import sys
import json
import time
import os
from typing import List, Optional, Dict
from datetime import datetime
import threading
from queue import Queue
from dataclasses import dataclass, asdict
import copy


@dataclass
class GPU:
    """Represents a GPU on a specific server."""
    server: str
    device_id: int
    user: Optional[str] = None
    port: int = 22
    ssh_key: Optional[str] = None

    def __str__(self):
        host = f"{self.user}@{self.server}" if self.user else self.server
        return f"{host}:gpu{self.device_id}"


@dataclass
class JobResult:
    """Results from a completed job."""
    job_id: int
    job_config: dict
    gpu: GPU
    returncode: int
    stdout: str
    stderr: str
    start_time: str
    end_time: str
    duration_seconds: float
    success: bool


class GPUWorker(threading.Thread):
    """Worker thread that processes jobs on a specific GPU."""

    def __init__(self, gpu: GPU, job_queue: Queue, results: List[JobResult],
                 script_path: str, tmux_session: str, tmux_pane: int,
                 verbose: bool = True):
        super().__init__()
        self.gpu = gpu
        self.job_queue = job_queue
        self.results = results
        self.script_path = script_path
        self.tmux_session = tmux_session
        self.tmux_pane = tmux_pane
        self.verbose = verbose
        self.daemon = True

    def run(self):
        """Process jobs from the queue."""
        while True:
            job = None
            try:
                job = self.job_queue.get(timeout=1)
                if job is None:  # Poison pill to stop thread
                    self.job_queue.task_done()
                    break

                job_id, job_config = job
                self.execute_job(job_id, job_config)
                self.job_queue.task_done()

            except Exception as e:
                # Ignore timeout exceptions when queue is empty
                from queue import Empty
                if not isinstance(e, Empty):
                    if self.verbose:
                        print(f"[{self.gpu}] Error: {e}", file=sys.stderr)
                if job is not None:
                    self.job_queue.task_done()

    def execute_job(self, job_id: int, job_config: dict):
        """Execute a single training job on this GPU."""
        start_time = datetime.now()

        if self.verbose:
            print(f"\n[{self.gpu}] Job {job_id} starting at {start_time.strftime('%H:%M:%S')}")
            print(f"[{self.gpu}] Config: {job_config}")

        # Copy base model to remote if needed
        base_model_hash = job_config.get('_base_model_hash')
        if base_model_hash:
            self.copy_base_model_to_remote(base_model_hash)

        # Build the training command with this GPU's device ID
        job_config = copy.deepcopy(job_config)
        job_config['device'] = self.gpu.device_id
        cmd = self.build_training_command(self.script_path, **job_config)

        # Prepend cd and source venv so it runs in the tmux pane
        full_cmd = f"cd ~/ml-customization && source env/bin/activate && {cmd}"

        # Execute via SSH
        returncode, stdout, stderr = self.run_ssh_command(full_cmd)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Copy results back to host if job succeeded
        experiment_path = None
        if returncode == 0:
            # For base_only jobs, copy base model instead of experiment results
            if job_config.get('mode') == 'base_only':
                experiment_path = self.copy_base_model_from_remote(job_config)
            else:
                experiment_path = self.copy_results_to_host(job_config)

        # Store result
        result = JobResult(
            job_id=job_id,
            job_config=job_config,
            gpu=self.gpu,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration_seconds=duration,
            success=returncode == 0
        )
        self.results.append(result)

        # Print completion
        status = "✓" if returncode == 0 else "✗"
        if self.verbose:
            print(f"[{self.gpu}] {status} Job {job_id} finished ({duration:.1f}s)")
            if returncode != 0 and stderr and stderr.strip():
                print(f"[{self.gpu}] Error: {stderr[:200]}")
            elif experiment_path:
                print(f"[{self.gpu}] Results copied to: {experiment_path}")

    def copy_base_model_to_remote(self, base_model_hash: str) -> bool:
        """
        Copy base model from local experiments/base_models/ to remote machine.

        Args:
            base_model_hash: Hash identifying the base model

        Returns:
            True if successful, False otherwise
        """
        local_base_dir = f"experiments/base_models/{base_model_hash}"
        remote_base_dir = f"~/ml-customization/experiments/base_models/{base_model_hash}"

        # Check if local base model exists
        if not os.path.exists(local_base_dir):
            if self.verbose:
                print(f"[{self.gpu}] Warning: Local base model not found: {local_base_dir}")
            return False

        # Check if remote already has this base model
        check_cmd = ["ssh", "-o", "StrictHostKeyChecking=no"]
        if self.gpu.port != 22:
            check_cmd.extend(["-p", str(self.gpu.port)])
        if self.gpu.ssh_key:
            check_cmd.extend(["-i", self.gpu.ssh_key])

        host_string = f"{self.gpu.user}@{self.gpu.server}" if self.gpu.user else self.gpu.server
        check_cmd.append(host_string)
        check_cmd.append(f"test -d {remote_base_dir}/best_base_model.pt")

        try:
            result = subprocess.run(check_cmd, capture_output=True, timeout=10)
            if result.returncode == 0:
                if self.verbose:
                    print(f"[{self.gpu}] Base model {base_model_hash} already cached on remote")
                return True
        except:
            pass

        # Copy base model to remote
        if self.verbose:
            print(f"[{self.gpu}] Copying base model {base_model_hash} to remote...")

        scp_cmd = ["scp", "-r", "-o", "StrictHostKeyChecking=no"]
        if self.gpu.port != 22:
            scp_cmd.extend(["-P", str(self.gpu.port)])
        if self.gpu.ssh_key:
            scp_cmd.extend(["-i", self.gpu.ssh_key])

        scp_cmd.append(local_base_dir)
        scp_cmd.append(f"{host_string}:~/ml-customization/experiments/base_models/")

        try:
            result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                if self.verbose:
                    print(f"[{self.gpu}] Successfully copied base model {base_model_hash}")
                return True
            else:
                if self.verbose:
                    print(f"[{self.gpu}] Failed to copy base model: {result.stderr}")
                return False
        except Exception as e:
            if self.verbose:
                print(f"[{self.gpu}] Error copying base model: {e}")
            return False

    def copy_base_model_from_remote(self, job_config: dict) -> Optional[str]:
        """
        Copy base model from remote machine to host after training.

        Args:
            job_config: Job configuration (must have '_base_model_hash')

        Returns:
            Local path where base model was copied, or None if failed
        """
        base_model_hash = job_config.get('_base_model_hash')
        if not base_model_hash:
            if self.verbose:
                print(f"[{self.gpu}] Warning: No base model hash in job config")
            return None

        remote_base_dir = f"~/ml-customization/experiments/base_models/{base_model_hash}"
        local_base_dir = f"experiments/base_models/{base_model_hash}"

        # Build scp command
        scp_cmd = ["scp", "-r", "-o", "StrictHostKeyChecking=no"]

        if self.gpu.port != 22:
            scp_cmd.extend(["-P", str(self.gpu.port)])

        if self.gpu.ssh_key:
            scp_cmd.extend(["-i", self.gpu.ssh_key])

        host_string = f"{self.gpu.user}@{self.gpu.server}" if self.gpu.user else self.gpu.server
        scp_cmd.append(f"{host_string}:{remote_base_dir}")
        scp_cmd.append(local_base_dir)

        try:
            # Create local base models directory if needed
            import os
            os.makedirs("experiments/base_models", exist_ok=True)

            # Copy files
            result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                if self.verbose:
                    print(f"[{self.gpu}] Base model copied to: {local_base_dir}")

                # Delete from remote machine to save space
                ssh_cmd = ["ssh", "-o", "StrictHostKeyChecking=no"]
                if self.gpu.port != 22:
                    ssh_cmd.extend(["-p", str(self.gpu.port)])
                if self.gpu.ssh_key:
                    ssh_cmd.extend(["-i", self.gpu.ssh_key])
                ssh_cmd.append(host_string)
                ssh_cmd.append(f"rm -rf {remote_base_dir}")

                subprocess.run(ssh_cmd, capture_output=True, timeout=30)

                return local_base_dir
            else:
                if self.verbose:
                    print(f"[{self.gpu}] Warning: Failed to copy base model: {result.stderr}")
                return None

        except Exception as e:
            if self.verbose:
                print(f"[{self.gpu}] Warning: Error copying base model: {e}")
            return None

    def copy_results_to_host(self, job_config: dict) -> Optional[str]:
        """
        Copy experiment results from remote machine to host and delete from remote.

        Returns:
            Local path where results were copied, or None if failed
        """
        # Determine the experiment directory path
        prefix = job_config.get('prefix', 'alpha')
        fold = job_config['fold']
        participants = job_config.get('participants', ['tonmoy', 'asfik', 'ejaz'])
        target_participant = participants[fold]

        remote_exp_dir = f"~/ml-customization/experiments/{prefix}/fold{fold}_{target_participant}"
        local_exp_dir = f"experiments/{prefix}/fold{fold}_{target_participant}"

        # Build scp command
        scp_cmd = ["scp", "-r", "-o", "StrictHostKeyChecking=no"]

        if self.gpu.port != 22:
            scp_cmd.extend(["-P", str(self.gpu.port)])

        if self.gpu.ssh_key:
            scp_cmd.extend(["-i", self.gpu.ssh_key])

        host_string = f"{self.gpu.user}@{self.gpu.server}" if self.gpu.user else self.gpu.server
        scp_cmd.append(f"{host_string}:{remote_exp_dir}")
        scp_cmd.append(local_exp_dir)

        try:
            # Create local experiments directory if needed
            import os
            os.makedirs(f"experiments/{prefix}", exist_ok=True)

            # Copy files
            result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                # Delete from remote machine
                ssh_cmd = ["ssh", "-o", "StrictHostKeyChecking=no"]
                if self.gpu.port != 22:
                    ssh_cmd.extend(["-p", str(self.gpu.port)])
                if self.gpu.ssh_key:
                    ssh_cmd.extend(["-i", self.gpu.ssh_key])
                ssh_cmd.append(host_string)
                ssh_cmd.append(f"rm -rf {remote_exp_dir}")

                subprocess.run(ssh_cmd, capture_output=True, timeout=30)

                return local_exp_dir
            else:
                if self.verbose:
                    print(f"[{self.gpu}] Warning: Failed to copy results: {result.stderr}")
                return None

        except Exception as e:
            if self.verbose:
                print(f"[{self.gpu}] Warning: Error copying results: {e}")
            return None

    def build_training_command(self, script_path: str, fold: int, device: int,
                               batch_size: int, model: str = 'medium',
                               prefix: str = 'alpha', participants: List[str] = None,
                               data_path: str = 'data/001_test',
                               use_augmentation: bool = False, lr: float = 3e-4,
                               early_stopping_patience: int = 40,
                               early_stopping_patience_target: int = 40,
                               mode: str = 'full_fine_tuning',
                               target_data_pct: float = 1.0,
                               n_base_participants: str = 'all',
                               window_size: int = 3000,
                               jitter_std: float = 0.005,
                               magnitude_range: List[float] = None,
                               aug_prob: float = 0.3,
                               **extra_args) -> str:
        """Build the training command with all arguments."""
        if participants is None:
            participants = ['tonmoy', 'asfik', 'ejaz']

        if magnitude_range is None:
            magnitude_range = [0.98, 1.02]

        cmd = f"python3 {script_path}"
        cmd += f" --fold {fold}"
        cmd += f" --device {device}"
        cmd += f" --batch_size {batch_size}"
        cmd += f" --model {model}"
        cmd += f" --prefix {prefix}"
        cmd += f" --lr {lr}"
        cmd += f" --early_stopping_patience {early_stopping_patience}"
        cmd += f" --early_stopping_patience_target {early_stopping_patience_target}"
        cmd += f" --mode {mode}"
        cmd += f" --target_data_pct {target_data_pct}"
        cmd += f" --n_base_participants {n_base_participants}"
        cmd += f" --data_path {data_path}"
        cmd += f" --window_size {window_size}"
        cmd += f" --participants {' '.join(participants)}"

        if use_augmentation:
            cmd += " --use_augmentation"
            cmd += f" --jitter_std {jitter_std}"
            cmd += f" --magnitude_range {magnitude_range[0]} {magnitude_range[1]}"
            cmd += f" --aug_prob {aug_prob}"

        # Add any extra arguments
        for key, value in extra_args.items():
            if isinstance(value, bool):
                if value:
                    cmd += f" --{key}"
            else:
                cmd += f" --{key} {value}"

        return cmd

    def run_ssh_command(self, command: str) -> tuple:
        """Execute command in tmux pane on remote host via SSH."""
        ssh_cmd = ["ssh", "-o", "StrictHostKeyChecking=no"]

        if self.gpu.port != 22:
            ssh_cmd.extend(["-p", str(self.gpu.port)])

        if self.gpu.ssh_key:
            ssh_cmd.extend(["-i", self.gpu.ssh_key])

        host_string = f"{self.gpu.user}@{self.gpu.server}" if self.gpu.user else self.gpu.server
        ssh_cmd.append(host_string)

        # Create a unique marker file to detect completion
        import uuid
        marker_file = f"/tmp/job_complete_{uuid.uuid4().hex}.marker"

        # Wrap command to run in tmux pane and capture exit status
        # Write exit code to marker file when done
        escaped_cmd = command.replace("'", "'\\''")  # Escape single quotes
        tmux_cmd = (
            f"tmux send-keys -t {self.tmux_session}.{self.tmux_pane} "
            f"'{escaped_cmd}; echo $? > {marker_file}' C-m && "
            f"while [ ! -f {marker_file} ]; do sleep 2; done"
        )

        ssh_cmd.append(tmux_cmd)

        # Execute and capture output
        try:
            result = subprocess.run(
                ssh_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=7200  # 2 hour timeout per job
            )

            # Read exit code from marker file
            read_exit_cmd = ["ssh", "-o", "StrictHostKeyChecking=no"]
            if self.gpu.port != 22:
                read_exit_cmd.extend(["-p", str(self.gpu.port)])
            if self.gpu.ssh_key:
                read_exit_cmd.extend(["-i", self.gpu.ssh_key])
            read_exit_cmd.append(host_string)
            read_exit_cmd.append(f"cat {marker_file} && rm -f {marker_file}")

            exit_result = subprocess.run(
                read_exit_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30
            )

            # Parse exit code
            exit_code = 0
            if exit_result.returncode == 0 and exit_result.stdout.strip().isdigit():
                exit_code = int(exit_result.stdout.strip())

            # Capture the output from tmux pane buffer
            capture_cmd = ["ssh", "-o", "StrictHostKeyChecking=no"]
            if self.gpu.port != 22:
                capture_cmd.extend(["-p", str(self.gpu.port)])
            if self.gpu.ssh_key:
                capture_cmd.extend(["-i", self.gpu.ssh_key])
            capture_cmd.append(host_string)
            capture_cmd.append(f"tmux capture-pane -t {self.tmux_session}.{self.tmux_pane} -p")

            capture_result = subprocess.run(
                capture_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30
            )

            stdout = capture_result.stdout if capture_result.returncode == 0 else result.stdout

            return exit_code, stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Job timeout (2 hours exceeded)"
        except Exception as e:
            return -1, "", str(e)


def setup_tmux_sessions(gpus: List[GPU], session_name: str = "ml_training") -> Dict[str, Dict[int, int]]:
    """
    Setup tmux sessions on each server with one pane per GPU.

    Returns:
        Dict mapping server -> {device_id: pane_index}
    """
    # Group GPUs by server
    servers = {}
    for gpu in gpus:
        server_key = f"{gpu.user}@{gpu.server}" if gpu.user else gpu.server
        if server_key not in servers:
            servers[server_key] = []
        servers[server_key].append(gpu)

    pane_mapping = {}

    for server_key, server_gpus in servers.items():
        # Get first GPU for SSH connection details
        first_gpu = server_gpus[0]

        # Build SSH command
        ssh_cmd = ["ssh", "-o", "StrictHostKeyChecking=no"]
        if first_gpu.port != 22:
            ssh_cmd.extend(["-p", str(first_gpu.port)])
        if first_gpu.ssh_key:
            ssh_cmd.extend(["-i", first_gpu.ssh_key])
        ssh_cmd.append(server_key)

        # Kill existing session if it exists, then create new one
        setup_cmd = f"tmux kill-session -t {session_name} 2>/dev/null; tmux new-session -d -s {session_name}"

        # Create additional panes for each GPU after the first
        for i in range(1, len(server_gpus)):
            setup_cmd += f" && tmux split-window -t {session_name}"

        # Select even-vertical layout for clean pane arrangement
        setup_cmd += f" && tmux select-layout -t {session_name} even-vertical"

        # Execute setup
        ssh_cmd.append(setup_cmd)

        try:
            result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"✓ Created tmux session '{session_name}' on {server_key} with {len(server_gpus)} panes")

                # Map device IDs to pane indices
                pane_mapping[first_gpu.server] = {gpu.device_id: i for i, gpu in enumerate(server_gpus)}
            else:
                print(f"✗ Failed to create tmux session on {server_key}: {result.stderr}")
                pane_mapping[first_gpu.server] = {}
        except Exception as e:
            print(f"✗ Error setting up tmux on {server_key}: {e}")
            pane_mapping[first_gpu.server] = {}

    return pane_mapping


def run_distributed_jobs(
    gpus: List[GPU],
    jobs: List[dict],
    script_path: str = "train.py",
    verbose: bool = True,
    log_file: Optional[str] = None,
    tmux_session: str = "ml_training"
) -> List[JobResult]:
    """
    Distribute jobs across multiple GPUs and servers using tmux.

    Args:
        gpus: List of GPU objects representing available resources
        jobs: List of job configurations
        script_path: Path to train.py on remote machines
        verbose: Print progress messages
        log_file: Optional path to save execution log
        tmux_session: Name of tmux session to create

    Returns:
        List of JobResult objects
    """
    start_time = datetime.now()

    print(f"\n{'='*80}")
    print(f"Distributed Training Job Manager")
    print(f"{'='*80}")
    print(f"Available GPUs: {len(gpus)}")
    for gpu in gpus:
        print(f"  - {gpu}")
    print(f"Total jobs: {len(jobs)}")
    print(f"Tmux session: {tmux_session}")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

    # Setup tmux sessions on each server
    print("Setting up tmux sessions...")
    pane_mapping = setup_tmux_sessions(gpus, tmux_session)
    print()

    # Create job queue and results list
    job_queue = Queue()
    results = []

    # Add all jobs to queue
    for i, job_config in enumerate(jobs):
        job_queue.put((i, job_config))

    # Create and start worker threads for each GPU
    workers = []
    for gpu in gpus:
        pane_idx = pane_mapping.get(gpu.server, {}).get(gpu.device_id, gpu.device_id)
        worker = GPUWorker(gpu, job_queue, results, script_path, tmux_session, pane_idx, verbose)
        worker.start()
        workers.append(worker)

    # Monitor progress
    total_jobs = len(jobs)
    if verbose:
        while not job_queue.empty():
            remaining = job_queue.qsize()
            completed = total_jobs - remaining
            print(f"\rProgress: {completed}/{total_jobs} jobs completed", end='', flush=True)
            time.sleep(2)
        print()  # New line after progress

    # Wait for all jobs to complete
    job_queue.join()

    # Stop workers
    for _ in workers:
        job_queue.put(None)
    for worker in workers:
        worker.join()

    # Print summary
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    successful = sum(1 for r in results if r.success)

    print(f"\n{'='*80}")
    print(f"Execution Summary")
    print(f"{'='*80}")
    print(f"Total jobs: {len(jobs)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")
    print(f"Total wall time: {total_duration:.1f}s ({total_duration/60:.1f}min)")

    # Calculate GPU utilization
    total_gpu_time = sum(r.duration_seconds for r in results)
    avg_gpu_utilization = (total_gpu_time / (total_duration * len(gpus))) * 100 if total_duration > 0 else 0
    print(f"GPU utilization: {avg_gpu_utilization:.1f}%")
    print(f"{'='*80}")

    # Print tmux connection info
    print(f"\nTmux Sessions:")
    servers_seen = set()
    for gpu in gpus:
        server_key = f"{gpu.user}@{gpu.server}" if gpu.user else gpu.server
        if server_key not in servers_seen:
            print(f"  ssh {server_key} -t 'tmux attach -t {tmux_session}'")
            servers_seen.add(server_key)
    print()

    # Print per-GPU summary
    print(f"\nPer-GPU Summary:")
    for gpu in gpus:
        gpu_results = [r for r in results if r.gpu == gpu]
        gpu_successful = sum(1 for r in gpu_results if r.success)
        gpu_time = sum(r.duration_seconds for r in gpu_results)
        print(f"  {gpu}: {len(gpu_results)} jobs, {gpu_successful} successful, {gpu_time:.1f}s total")

    # Show failed jobs
    failed = [r for r in results if not r.success]
    if failed:
        print(f"\nFailed Jobs:")
        for r in failed:
            print(f"  Job {r.job_id} on {r.gpu}: exit code {r.returncode}")
            if r.stderr:
                print(f"    Error: {r.stderr[:100]}")

    print()

    # Save log
    if log_file:
        log_data = {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'total_duration_seconds': total_duration,
            'gpus': [str(gpu) for gpu in gpus],
            'total_jobs': len(jobs),
            'successful_jobs': successful,
            'failed_jobs': len(results) - successful,
            'gpu_utilization_percent': avg_gpu_utilization,
            'results': [
                {
                    'job_id': r.job_id,
                    'gpu': str(r.gpu),
                    'config': r.job_config,
                    'success': r.success,
                    'returncode': r.returncode,
                    'start_time': r.start_time,
                    'end_time': r.end_time,
                    'duration_seconds': r.duration_seconds,
                    'stderr': r.stderr if not r.success else None
                }
                for r in results
            ]
        }
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        print(f"Execution log saved to: {log_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Distribute training jobs across multiple servers and GPUs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:

  # Run jobs using cluster config file
  python run_distributed_training.py \\
      --cluster-config cluster.json \\
      --jobs-config jobs.json \\
      --log-file results.json

  # cluster.json format:
  {
    "servers": [
      {"host": "server1.example.com", "user": "myuser", "gpus": 2},
      {"host": "server2.example.com", "user": "myuser", "gpus": 1, "ssh_key": "~/.ssh/id_rsa"},
      {"host": "192.168.1.100", "user": "myuser", "gpus": 2, "port": 2222}
    ],
    "script_path": "train.py"
  }

  Note: Script runs from ~/ml-customization directory on remote machines

  # jobs.json format:
  [
    {"fold": 0, "batch_size": 64, "prefix": "exp1"},
    {"fold": 1, "batch_size": 64, "prefix": "exp1"},
    {"fold": 2, "batch_size": 64, "prefix": "exp1"}
  ]

  Note: 'device' field in job configs is automatically set based on GPU assignment
        """
    )

    parser.add_argument('--cluster-config', required=True,
                       help='Path to JSON file with cluster configuration')
    parser.add_argument('--jobs-config', required=True,
                       help='Path to JSON file with job configurations')
    parser.add_argument('--script-path', default='train.py',
                       help='Path to train.py relative to ~/ml-customization (can be overridden in cluster config)')
    parser.add_argument('--log-file', help='Path to save execution log (JSON format)')
    parser.add_argument('--tmux-session', default='ml_training',
                       help='Name of tmux session to create (default: ml_training)')
    parser.add_argument('--quiet', action='store_true', help='Reduce output verbosity')

    args = parser.parse_args()

    # Load cluster configuration
    with open(args.cluster_config) as f:
        cluster_config = json.load(f)

    # Build list of GPUs
    gpus = []
    for server in cluster_config['servers']:
        num_gpus = server['gpus']
        for device_id in range(num_gpus):
            gpu = GPU(
                server=server['host'],
                device_id=device_id,
                user=server.get('user'),
                port=server.get('port', 22),
                ssh_key=server.get('ssh_key')
            )
            gpus.append(gpu)

    # Load job configurations
    with open(args.jobs_config) as f:
        jobs = json.load(f)

    # Get script path (from cluster config or command line)
    script_path = cluster_config.get('script_path', args.script_path)

    # Run distributed jobs
    run_distributed_jobs(
        gpus=gpus,
        jobs=jobs,
        script_path=script_path,
        verbose=not args.quiet,
        log_file=args.log_file,
        tmux_session=args.tmux_session
    )


if __name__ == '__main__':
    main()
