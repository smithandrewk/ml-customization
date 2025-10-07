#!/usr/bin/env python3
"""
Extract and re-run specific failed fine-tuning jobs.
"""

import json
import os
import shutil
import argparse

# Failed jobs: (prefix, fold)
FAILED_JOBS = [
    ("finetune_0cd2204ea7c1ca8a", 4),  # fold4_ejaz
    ("finetune_16da73e1e45d3170", 6),  # fold6_unk1
    ("finetune_1775ba93f2b9bf75", 1),  # fold1_asfik
    ("finetune_1775ba93f2b9bf75", 4),  # fold4_ejaz
    ("finetune_1775ba93f2b9bf75", 6),  # fold6_unk1
    ("finetune_24c53bf385240ec9", 4),  # fold4_ejaz
    ("finetune_27973c3b012ace93", 6),  # fold6_unk1
    ("target_only_2e36a01dc507719e", 3),  # fold3_anam
]


def main():
    parser = argparse.ArgumentParser(description="Re-run failed fine-tuning jobs")
    parser.add_argument("--jobs-config", default="finetune_jobs.json",
                       help="Original fine-tuning jobs config")
    parser.add_argument("--output", default="failed_jobs.json",
                       help="Output file for failed jobs config")
    parser.add_argument("--delete-dirs", action="store_true",
                       help="Delete incomplete experiment directories")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without doing it")
    args = parser.parse_args()

    # Load original jobs config
    with open(args.jobs_config) as f:
        all_jobs = json.load(f)

    # Filter to just failed jobs
    failed_jobs_list = []
    for job in all_jobs:
        prefix = job["prefix"]
        fold = job["fold"]
        if (prefix, fold) in FAILED_JOBS:
            failed_jobs_list.append(job)
            print(f"✓ Found failed job: {prefix}/fold{fold}")

    print(f"\nFound {len(failed_jobs_list)} failed jobs to re-run")

    # Write filtered config
    if not args.dry_run:
        with open(args.output, 'w') as f:
            json.dump(failed_jobs_list, f, indent=2)
        print(f"✓ Wrote failed jobs config to: {args.output}")
    else:
        print(f"[DRY RUN] Would write to: {args.output}")

    # Optionally delete incomplete experiment directories
    if args.delete_dirs:
        print("\nDeleting incomplete experiment directories:")
        participants = all_jobs[0]["participants"]  # Assuming same for all

        for prefix, fold in FAILED_JOBS:
            target_participant = participants[fold]
            exp_dir = f"experiments/{prefix}/fold{fold}_{target_participant}"

            if os.path.exists(exp_dir):
                if args.dry_run:
                    print(f"[DRY RUN] Would delete: {exp_dir}")
                else:
                    shutil.rmtree(exp_dir)
                    print(f"✓ Deleted: {exp_dir}")
            else:
                print(f"  (not found): {exp_dir}")

    # Print next steps
    print("\n" + "="*80)
    print("Next steps:")
    print("="*80)
    if args.dry_run:
        print("1. Run without --dry-run to actually create the config")
    else:
        print(f"Run the following command to re-run failed jobs:\n")
        print(f"  python3 run_distributed_training.py \\")
        print(f"    --cluster-config cluster_config.json \\")
        print(f"    --jobs-config {args.output} \\")
        print(f"    --script-path train_finetune.py \\")
        print(f"    --tmux-session ml_training \\")
        print(f"    --log-file failed_jobs_log.json")
        print()


if __name__ == "__main__":
    main()
