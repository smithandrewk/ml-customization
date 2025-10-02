#!/usr/bin/env python3
"""
Sync code to cluster servers and clean experiment directories.
Runs git pull and optionally removes experiments directory on all servers.
"""

import argparse
import subprocess
import json
import sys
from typing import List, Dict


def run_command_on_server(server: Dict[str, any], command: str, verbose: bool = True) -> tuple:
    """Run a command on a remote server via SSH."""
    ssh_cmd = ["ssh", "-o", "StrictHostKeyChecking=no"]

    if server.get('port', 22) != 22:
        ssh_cmd.extend(["-p", str(server['port'])])

    if server.get('ssh_key'):
        ssh_cmd.extend(["-i", server['ssh_key']])

    host_string = f"{server.get('user')}@{server['host']}" if server.get('user') else server['host']
    ssh_cmd.append(host_string)
    ssh_cmd.append(command)

    if verbose:
        print(f"[{host_string}] Running: {command}")

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


def sync_cluster(cluster_config_path: str, clean_experiments: bool = False,
                git_pull: bool = True, push_first: bool = False, verbose: bool = True):
    """
    Sync code to all servers in cluster and optionally clean experiments.

    Args:
        cluster_config_path: Path to cluster_config.json
        clean_experiments: If True, remove experiments directory on each server
        git_pull: If True, run git pull on each server
        push_first: If True, push local changes to origin first
        verbose: Print detailed output
    """
    # Load cluster configuration
    with open(cluster_config_path) as f:
        cluster_config = json.load(f)

    servers = cluster_config['servers']

    print(f"\n{'='*80}")
    print(f"Cluster Sync")
    print(f"{'='*80}")
    print(f"Servers: {len(servers)}")
    print(f"Git pull: {git_pull}")
    print(f"Push first: {push_first}")
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
        print(f"\n--- {host_string} ---")

        commands = []

        # Navigate to directory
        base_cmd = "cd ~/ml-customization"

        # Git pull
        if git_pull:
            # First checkout main branch
            checkout_cmd = f"{base_cmd} && git fetch origin && git checkout main"
            returncode, stdout, stderr = run_command_on_server(server, checkout_cmd, verbose)

            if returncode == 0:
                print(f"[{host_string}] ✓ Checked out main")
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

                # Fetch and reset to main
                reset_cmd = f"{base_cmd} && git fetch origin && git reset --hard origin/main"
                returncode, stdout, stderr = run_command_on_server(server, reset_cmd, verbose)

                if returncode == 0:
                    print(f"[{host_string}] ✓ Git fetch and reset successful")
                else:
                    print(f"[{host_string}] ✗ Git sync failed - continuing anyway")
                    # Don't fail completely, just warn
                    print(f"[{host_string}] ⚠ Warning: Code may be out of sync")

        # Clean experiments directory
        if clean_experiments:
            clean_cmd = f"{base_cmd} && rm -rf experiments && mkdir -p experiments"
            returncode, stdout, stderr = run_command_on_server(server, clean_cmd, verbose)

            if returncode == 0:
                print(f"[{host_string}] ✓ Experiments directory cleaned")
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

  # Only clean experiments (no git pull)
  python sync_cluster.py --cluster-config cluster_config.json --clean --no-git-pull

  # Quiet mode (less verbose output)
  python sync_cluster.py --cluster-config cluster_config.json --clean --quiet
        """
    )

    parser.add_argument('--cluster-config', required=False, default='cluster_config.json',
                       help='Path to cluster configuration JSON file')
    parser.add_argument('--clean', action='store_true',
                       help='Remove experiments directory on all servers')
    parser.add_argument('--no-git-pull', action='store_true',
                       help='Skip git pull (only used with --clean)')
    parser.add_argument('--push-first', action='store_true',
                       help='Push local changes to origin before syncing servers')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')

    args = parser.parse_args()

    if args.no_git_pull and not args.clean:
        print("Warning: --no-git-pull has no effect without --clean", file=sys.stderr)

    success = sync_cluster(
        cluster_config_path=args.cluster_config,
        clean_experiments=args.clean,
        git_pull=not args.no_git_pull,
        push_first=args.push_first,
        verbose=not args.quiet
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
