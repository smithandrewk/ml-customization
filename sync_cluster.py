#!/usr/bin/env python3
"""
Sync code and data to cluster servers and clean experiment directories.
- Runs git pull on all servers
- Syncs data files using rsync (only copies files that don't exist on remote)
- Optionally removes experiments directory on all servers
"""

import argparse
import subprocess
import json
import sys
import socket
from typing import List, Dict


def is_localhost(server: Dict[str, any]) -> bool:
    """Check if server refers to localhost."""
    local_hostname = socket.gethostname()
    local_fqdn = socket.getfqdn()

    localhost_names = [
        'localhost',
        '127.0.0.1',
        '::1',
        local_hostname,
        local_fqdn,
    ]

    return server['host'].lower() in [name.lower() for name in localhost_names]


def run_command_on_server(server: Dict[str, any], command: str, verbose: bool = True) -> tuple:
    """Run a command on a remote server via SSH (or locally if localhost)."""
    host_string = f"{server.get('user')}@{server['host']}" if server.get('user') else server['host']

    if verbose:
        print(f"[{host_string}] Running: {command}")

    # If localhost, run directly without SSH
    if is_localhost(server):
        try:
            result = subprocess.run(
                ["bash", "-c", command],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=60,
                cwd=None  # Run in current directory
            )

            if verbose:
                if result.stdout:
                    print(f"[{host_string}] {result.stdout.strip()}")
                if result.returncode != 0 and result.stderr:
                    print(f"[{host_string}] Error: {result.stderr.strip()}", file=sys.stderr)

            return result.returncode, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            if verbose:
                print(f"[{host_string}] Command timeout", file=sys.stderr)
            return -1, "", "Command timeout"
        except Exception as e:
            if verbose:
                print(f"[{host_string}] Error: {e}", file=sys.stderr)
            return -1, "", str(e)

    # Remote server - use SSH
    ssh_cmd = ["ssh", "-o", "StrictHostKeyChecking=no"]

    if server.get('port', 22) != 22:
        ssh_cmd.extend(["-p", str(server['port'])])

    if server.get('ssh_key'):
        ssh_cmd.extend(["-i", server['ssh_key']])

    ssh_cmd.append(host_string)
    ssh_cmd.append(command)

    try:
        result = subprocess.run(
            ssh_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60
        )

        if verbose:
            if result.stdout:
                print(f"[{host_string}] {result.stdout.strip()}")
            if result.returncode != 0 and result.stderr:
                print(f"[{host_string}] Error: {result.stderr.strip()}", file=sys.stderr)

        return result.returncode, result.stdout, result.stderr

    except subprocess.TimeoutExpired:
        if verbose:
            print(f"[{host_string}] Command timeout", file=sys.stderr)
        return -1, "", "Command timeout"
    except Exception as e:
        if verbose:
            print(f"[{host_string}] Error: {e}", file=sys.stderr)
        return -1, "", str(e)


def sync_data_to_server(server: Dict[str, any], source_path: str,
                        verbose: bool = True, dry_run: bool = False) -> bool:
    """
    Sync data files to a remote server using rsync with --ignore-existing.
    Only copies files that don't already exist on the remote.
    Maintains the same directory structure on remote as local.

    Args:
        server: Server dict from cluster config
        source_path: Local path to sync from (e.g., 'data/003_tmp2')
        verbose: Show progress
        dry_run: Show what would be synced without actually copying

    Returns:
        bool: True if successful, False otherwise
    """
    host_string = f"{server.get('user')}@{server['host']}" if server.get('user') else server['host']

    # Skip localhost
    if is_localhost(server):
        if verbose:
            print(f"[{host_string}] Skipping (localhost - already has the data)")
        return True

    # Build rsync command
    rsync_cmd = ['rsync', '-avz', '--progress']

    # Add dry-run flag if requested
    if dry_run:
        rsync_cmd.append('--dry-run')

    # Only copy files that don't exist on destination
    rsync_cmd.append('--ignore-existing')

    # Exclude common directories that shouldn't be synced
    excludes = [
        'env/',
        'venv/',
        '.venv/',
        '.git/',
        '__pycache__/',
        '*.pyc',
        '.pytest_cache/',
        '.ipynb_checkpoints/',
        '*.log',
        'experiments/',  # Don't sync experiments directory
    ]

    for exclude in excludes:
        rsync_cmd.extend(['--exclude', exclude])

    # Add SSH options
    ssh_opts = '-o StrictHostKeyChecking=no'
    if server.get('port', 22) != 22:
        ssh_opts += f' -p {server["port"]}'
    if server.get('ssh_key'):
        ssh_opts += f' -i {server["ssh_key"]}'

    rsync_cmd.extend(['-e', f'ssh {ssh_opts}'])

    # Use --relative to preserve directory structure
    # If source is 'data/003_tmp2', it will create 'data/003_tmp2' on remote
    rsync_cmd.append('--relative')

    # Clean the source path - strip any whitespace/newlines and trailing slashes
    source = source_path.strip().rstrip('/')
    rsync_cmd.append(source)

    # Destination is the ml-customization directory on remote
    rsync_cmd.append(f"{host_string}:~/ml-customization/")

    if verbose:
        print(f"[{host_string}] Syncing: {source} -> ~/ml-customization/")
        print(f"[{host_string}] Running rsync (only new files)...")

    # Execute rsync
    try:
        result = subprocess.run(
            rsync_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=1800  # 30 minute timeout
        )

        if result.returncode == 0:
            if verbose:
                # Show summary of what was transferred
                lines = result.stdout.split('\n')
                transferred_files = [l for l in lines if l and not l.startswith('sending') and not l.startswith('total')]
                if transferred_files and len(transferred_files) > 5:
                    print(f"[{host_string}] ✓ Data sync complete ({len(transferred_files)} files)")
                elif transferred_files:
                    print(f"[{host_string}] ✓ Data sync complete")
                else:
                    print(f"[{host_string}] ✓ Data sync complete (no new files to copy)")
            return True
        else:
            print(f"[{host_string}] ✗ Failed to sync data", file=sys.stderr)
            if result.stderr:
                print(f"[{host_string}] Error: {result.stderr[:200]}", file=sys.stderr)
            return False

    except subprocess.TimeoutExpired:
        print(f"[{host_string}] ✗ Data sync timed out after 30 minutes", file=sys.stderr)
        return False
    except Exception as e:
        print(f"[{host_string}] ✗ Error syncing data: {e}", file=sys.stderr)
        return False


def sync_cluster(cluster_config_path: str, clean_experiments: bool = False,
                git_pull: bool = True, push_first: bool = False, branch: str = "main",
                sync_data: bool = False, data_path: str = None,
                verbose: bool = True):
    """
    Sync code to all servers in cluster and optionally clean experiments.

    Args:
        cluster_config_path: Path to cluster_config.json
        clean_experiments: If True, remove experiments directory on each server
        git_pull: If True, run git pull on each server
        push_first: If True, push local changes to origin first
        branch: Git branch to checkout (default: main)
        sync_data: If True, rsync data files to cluster (only new files)
        data_path: Local directory to sync (required if sync_data is True)
        verbose: Print detailed output
    """
    # Load cluster configuration
    with open(cluster_config_path) as f:
        cluster_config = json.load(f)

    servers = cluster_config['servers']

    # Validate that data_path is provided if sync_data is True
    if sync_data and data_path is None:
        print("Error: --data-path is required when using --sync-data", file=sys.stderr)
        return False

    print(f"\n{'='*80}")
    print(f"Cluster Sync")
    print(f"{'='*80}")
    print(f"Servers: {len(servers)}")
    print(f"Branch: {branch}")
    print(f"Git pull: {git_pull}")
    print(f"Push first: {push_first}")
    print(f"Sync data: {sync_data}")
    if sync_data:
        print(f"Data path: {data_path}")
    print(f"Clean experiments: {clean_experiments}")
    print(f"{'='*80}\n")

    # Push local changes first if requested
    if push_first:
        print("--- Pushing local changes to origin ---")
        try:
            result = subprocess.run(
                ["git", "push"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                print("✓ Pushed local changes to origin")
                if result.stdout:
                    print(result.stdout)
            else:
                print(f"✗ Failed to push: {result.stderr}")
                print("Continuing anyway...")
        except Exception as e:
            print(f"✗ Error pushing: {e}")
            print("Continuing anyway...")
        print()

    success_count = 0
    fail_count = 0

    for server in servers:
        host_string = f"{server.get('user')}@{server['host']}" if server.get('user') else server['host']

        # Mark localhost for clarity
        if is_localhost(server):
            print(f"\n--- {host_string} (localhost) ---")
        else:
            print(f"\n--- {host_string} ---")

        commands = []

        # Navigate to directory
        base_cmd = "cd ~/ml-customization"

        # Git pull
        if git_pull:
            # First checkout specified branch
            checkout_cmd = f"{base_cmd} && git fetch origin && git checkout {branch}"
            returncode, stdout, stderr = run_command_on_server(server, checkout_cmd, verbose)

            if returncode == 0:
                print(f"[{host_string}] ✓ Checked out {branch}")
            else:
                print(f"[{host_string}] ✗ Failed to checkout branch - continuing anyway")
                print(f"[{host_string}] ⚠ Warning: May be on wrong branch")

            # Now pull latest changes
            git_cmd = f"{base_cmd} && git pull"
            returncode, stdout, stderr = run_command_on_server(server, git_cmd, verbose)

            if returncode == 0:
                print(f"[{host_string}] ✓ Git pull successful")
            else:
                # If git pull fails, try to fetch and reset to origin
                print(f"[{host_string}] Regular git pull failed, trying fetch and reset...")

                # Fetch and reset to specified branch
                reset_cmd = f"{base_cmd} && git fetch origin && git reset --hard origin/{branch}"
                returncode, stdout, stderr = run_command_on_server(server, reset_cmd, verbose)

                if returncode == 0:
                    print(f"[{host_string}] ✓ Git fetch and reset successful")
                else:
                    print(f"[{host_string}] ✗ Git sync failed - continuing anyway")
                    # Don't fail completely, just warn
                    print(f"[{host_string}] ⚠ Warning: Code may be out of sync")

            # Always clear Python cache after git sync to avoid stale bytecode
            cache_cmd = f"{base_cmd} && find . -type d -name __pycache__ -exec rm -rf {{}} + 2>/dev/null; true"
            returncode, stdout, stderr = run_command_on_server(server, cache_cmd, verbose=False)
            if returncode == 0:
                print(f"[{host_string}] ✓ Python cache cleared")

            # Install/update Python dependencies
            pip_cmd = f"{base_cmd} && source env/bin/activate && pip install -q -r requirements.txt"
            returncode, stdout, stderr = run_command_on_server(server, pip_cmd, verbose=False)

            if returncode == 0:
                print(f"[{host_string}] ✓ Dependencies installed")
            else:
                print(f"[{host_string}] ✗ Failed to install dependencies - continuing anyway")
                print(f"[{host_string}] ⚠ Warning: Dependencies may be out of date")

        # Sync data files using rsync
        if sync_data:
            data_success = sync_data_to_server(server, data_path, verbose=verbose)
            if not data_success:
                print(f"[{host_string}] ⚠ Warning: Data sync failed")

        # Clean experiments directory and Python cache
        if clean_experiments:
            clean_cmd = f"{base_cmd} && rm -rf experiments && mkdir -p experiments && find . -type d -name __pycache__ -exec rm -rf {{}} + 2>/dev/null; true"
            returncode, stdout, stderr = run_command_on_server(server, clean_cmd, verbose)

            if returncode == 0:
                print(f"[{host_string}] ✓ Experiments directory and Python cache cleaned")
            else:
                print(f"[{host_string}] ✗ Failed to clean experiments directory")
                fail_count += 1
                continue

        success_count += 1

    print(f"\n{'='*80}")
    print(f"Sync Summary")
    print(f"{'='*80}")
    print(f"Successful: {success_count}/{len(servers)}")
    print(f"Failed: {fail_count}/{len(servers)}")
    print(f"{'='*80}\n")

    return success_count == len(servers)


def main():
    parser = argparse.ArgumentParser(
        description="Sync code and clean experiments on cluster servers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Git pull on all servers
  python sync_cluster.py --cluster-config cluster_config.json

  # Git pull and clean experiments directory
  python sync_cluster.py --cluster-config cluster_config.json --clean

  # Sync data files to cluster (only copies new files)
  python sync_cluster.py --sync-data --data-path ./data

  # Sync data without git operations
  python sync_cluster.py --sync-data --data-path ./data --no-git-pull

  # Only clean experiments (no git pull)
  python sync_cluster.py --cluster-config cluster_config.json --clean --no-git-pull

  # Quiet mode (less verbose output)
  python sync_cluster.py --cluster-config cluster_config.json --clean --quiet
        """
    )

    parser.add_argument('--cluster-config', required=False, default='configs/cluster_config.json',
                       help='Path to cluster configuration JSON file')
    parser.add_argument('--clean', action='store_true',
                       help='Remove experiments directory on all servers')
    parser.add_argument('--no-git-pull', action='store_true',
                       help='Skip git pull (only used with --clean)')
    parser.add_argument('--push-first', action='store_true',
                       help='Push local changes to origin before syncing servers')
    parser.add_argument('--branch', default='main',
                       help='Git branch to checkout on servers (default: main)')
    parser.add_argument('--sync-data', action='store_true',
                       help='Sync data files to cluster using rsync (only copies new files, requires --data-path)')
    parser.add_argument('--data-path',
                       help='Local directory to sync (required when using --sync-data)')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')

    args = parser.parse_args()

    if args.no_git_pull and not args.clean:
        print("Warning: --no-git-pull has no effect without --clean", file=sys.stderr)

    if args.sync_data and not args.data_path:
        parser.error("--data-path is required when using --sync-data")

    success = sync_cluster(
        cluster_config_path=args.cluster_config,
        clean_experiments=args.clean,
        git_pull=not args.no_git_pull,
        push_first=args.push_first,
        branch=args.branch,
        sync_data=args.sync_data,
        data_path=args.data_path,
        verbose=not args.quiet
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
