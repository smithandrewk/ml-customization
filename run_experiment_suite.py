#!/usr/bin/env python3
"""
Comprehensive Experiment Suite for Advanced Customization Techniques Ablation Study

This script runs systematic experiments to evaluate the impact of each advanced 
customization technique individually and in combination.

Usage:
    python run_experiment_suite.py --dataset_dir data/001_tonmoy_60s --participants tonmoy,asfik,ejaz
    python run_experiment_suite.py --config experiments/experiment_3.yaml --dataset_dir data/001_tonmoy_60s --participants all
"""

import os
import sys
import argparse
import subprocess
import time
import yaml
import itertools
from pathlib import Path
import json
from datetime import datetime

# Experiment configurations
EXPERIMENT_CONFIGS = {
    # Baseline
    'baseline': {
        'name': 'Baseline',
        'description': 'Standard two-phase customization (no advanced techniques)',
        'flags': []
    },
    
    # Individual techniques
    'ewc_only': {
        'name': 'EWC Only',
        'description': 'Elastic Weight Consolidation regularization only',
        'flags': ['--use_ewc']
    },
    'layerwise_only': {
        'name': 'Layer-wise Only', 
        'description': 'Layer-wise fine-tuning only',
        'flags': ['--use_layerwise_finetuning']
    },
    'gradual_only': {
        'name': 'Gradual Unfreezing Only',
        'description': 'Gradual unfreezing only',
        'flags': ['--use_gradual_unfreezing']
    },
    'augmentation_only': {
        'name': 'Augmentation Only',
        'description': 'Time-series data augmentation only',
        'flags': ['--use_augmentation']
    },
    'coral_only': {
        'name': 'CORAL Only',
        'description': 'CORAL domain adaptation only',
        'flags': ['--use_coral']
    },
    'contrastive_only': {
        'name': 'Contrastive Only',
        'description': 'Contrastive learning only',
        'flags': ['--use_contrastive']
    },
    'ensemble_only': {
        'name': 'Ensemble Only',
        'description': 'Ensemble approach only',
        'flags': ['--use_ensemble']
    },
    
    # Promising combinations
    'ewc_layerwise': {
        'name': 'EWC + Layer-wise',
        'description': 'Combine EWC regularization with layer-wise fine-tuning',
        'flags': ['--use_ewc', '--use_layerwise_finetuning']
    },
    'ewc_augmentation': {
        'name': 'EWC + Augmentation',
        'description': 'Combine EWC with data augmentation',
        'flags': ['--use_ewc', '--use_augmentation']
    },
    'layerwise_augmentation': {
        'name': 'Layer-wise + Augmentation',
        'description': 'Combine layer-wise fine-tuning with data augmentation',
        'flags': ['--use_layerwise_finetuning', '--use_augmentation']
    },
    'core_trio': {
        'name': 'Core Trio',
        'description': 'EWC + Layer-wise + Augmentation (most promising combination)',
        'flags': ['--use_ewc', '--use_layerwise_finetuning', '--use_augmentation']
    },
    'domain_adaptation': {
        'name': 'Domain Adaptation',
        'description': 'CORAL + Contrastive learning for domain alignment',
        'flags': ['--use_coral', '--use_contrastive']
    },
    'all_advanced': {
        'name': 'All Advanced',
        'description': 'All advanced techniques combined',
        'flags': ['--use_ewc', '--use_layerwise_finetuning', '--use_gradual_unfreezing', 
                 '--use_augmentation', '--use_coral', '--use_contrastive']
    },
    'all_plus_ensemble': {
        'name': 'All + Ensemble',
        'description': 'All techniques including ensemble',
        'flags': ['--use_ewc', '--use_layerwise_finetuning', '--use_gradual_unfreezing',
                 '--use_augmentation', '--use_coral', '--use_contrastive', '--use_ensemble']
    }
}

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run comprehensive ablation study experiments')
    parser.add_argument('--config', type=str, default='configs/config.yaml', 
                       help='Base config file to use')
    parser.add_argument('--dataset_dir', type=str, required=True,
                       help='Directory containing participant data')
    parser.add_argument('--participants', type=str, required=True,
                       help='Comma-separated list of participants, or "all" for all available')
    parser.add_argument('--experiments', type=str, 
                       help='Comma-separated list of experiments to run (default: all)')
    parser.add_argument('--output_dir', type=str, default='ablation_study_results',
                       help='Directory to store experiment results')
    parser.add_argument('--dry_run', action='store_true',
                       help='Print commands without executing them')
    parser.add_argument('--parallel', type=int, default=1,
                       help='Number of parallel experiments (default: 1 for sequential)')
    parser.add_argument('--model', type=str, default='simple', choices=['full', 'simple'],
                       help='Model architecture to use (simple for faster experiments)')
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='Maximum epochs per experiment (to speed up ablation study)')
    
    return parser.parse_args()

def get_available_participants(dataset_dir):
    """Get list of available participants from dataset directory."""
    participants = []
    for file in os.listdir(dataset_dir):
        if file.endswith('_train.pt'):
            participant = file.replace('_train.pt', '')
            participants.append(participant)
    return sorted(participants)

def create_output_structure(output_dir):
    """Create organized output directory structure."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_dir = Path(output_dir) / f"ablation_study_{timestamp}"
    
    # Create subdirectories
    (study_dir / "experiments").mkdir(parents=True, exist_ok=True)
    (study_dir / "logs").mkdir(parents=True, exist_ok=True)
    (study_dir / "analysis").mkdir(parents=True, exist_ok=True)
    
    return study_dir

def run_single_experiment(config_file, dataset_dir, participant, experiment_config, 
                         study_dir, model, max_epochs, dry_run=False):
    """Run a single experiment for one participant with one technique configuration."""
    
    exp_name = experiment_config['name'].replace(' ', '_').replace('+', 'plus').lower()
    exp_dir = study_dir / "experiments" / f"{participant}_{exp_name}"
    log_file = study_dir / "logs" / f"{participant}_{exp_name}.log"
    
    # Build command
    cmd = [
        'python', 'train.py',
        '--config', str(config_file),
        '--dataset_dir', str(dataset_dir),
        '--target_participant', participant,
        '--model', model,
        '--experiment_suffix', f"ablation_{exp_name}"
    ]
    
    # Add technique flags
    cmd.extend(experiment_config['flags'])
    
    # Add epoch limit for faster experiments
    # Note: This would require adding --max_epochs argument to train.py
    # For now, we'll rely on early stopping
    
    cmd_str = ' '.join(cmd)
    
    print(f"🧪 Running: {participant} - {experiment_config['name']}")
    print(f"   Command: {cmd_str}")
    
    if dry_run:
        print(f"   [DRY RUN] Would save to: {exp_dir}")
        return True
    
    try:
        # Run experiment with logging
        with open(log_file, 'w') as log:
            log.write(f"Experiment: {experiment_config['name']}\n")
            log.write(f"Participant: {participant}\n")
            log.write(f"Command: {cmd_str}\n")
            log.write(f"Started: {datetime.now()}\n")
            log.write("-" * 60 + "\n")
            log.flush()
            
            result = subprocess.run(
                cmd, 
                stdout=log, 
                stderr=subprocess.STDOUT,
                text=True,
                cwd=os.getcwd()
            )
            
            log.write(f"\nCompleted: {datetime.now()}\n")
            log.write(f"Return code: {result.returncode}\n")
        
        if result.returncode == 0:
            print(f"   ✅ Success: {participant} - {experiment_config['name']}")
            return True
        else:
            print(f"   ❌ Failed: {participant} - {experiment_config['name']} (code: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"   💥 Error: {participant} - {experiment_config['name']}: {str(e)}")
        return False

def save_experiment_metadata(study_dir, args, participants, experiments_to_run):
    """Save metadata about the experiment suite."""
    metadata = {
        'study_info': {
            'timestamp': datetime.now().isoformat(),
            'dataset_dir': str(args.dataset_dir),
            'base_config': str(args.config),
            'model_architecture': args.model,
            'max_epochs': args.max_epochs
        },
        'participants': participants,
        'experiments': {
            exp_key: {
                'name': exp_config['name'],
                'description': exp_config['description'],
                'flags': exp_config['flags']
            }
            for exp_key, exp_config in experiments_to_run.items()
        },
        'execution_plan': {
            'total_experiments': len(participants) * len(experiments_to_run),
            'estimated_duration_hours': len(participants) * len(experiments_to_run) * 0.5,  # Rough estimate
            'parallel_jobs': args.parallel
        }
    }
    
    metadata_file = study_dir / "experiment_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📋 Experiment metadata saved: {metadata_file}")
    return metadata

def main():
    args = parse_arguments()
    
    # Validate inputs
    if not os.path.exists(args.dataset_dir):
        print(f"❌ Dataset directory not found: {args.dataset_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)
    
    # Get participants
    if args.participants.lower() == 'all':
        participants = get_available_participants(args.dataset_dir)
    else:
        participants = [p.strip() for p in args.participants.split(',')]
    
    # Validate participants
    available_participants = get_available_participants(args.dataset_dir)
    for p in participants:
        if p not in available_participants:
            print(f"❌ Participant '{p}' not found in dataset. Available: {available_participants}")
            sys.exit(1)
    
    # Select experiments to run
    if args.experiments:
        experiment_keys = [e.strip() for e in args.experiments.split(',')]
        experiments_to_run = {k: EXPERIMENT_CONFIGS[k] for k in experiment_keys if k in EXPERIMENT_CONFIGS}
    else:
        experiments_to_run = EXPERIMENT_CONFIGS
    
    # Create output structure
    study_dir = create_output_structure(args.output_dir)
    
    # Save metadata
    metadata = save_experiment_metadata(study_dir, args, participants, experiments_to_run)
    
    print(f"\n🚀 ABLATION STUDY LAUNCH")
    print(f"📁 Results directory: {study_dir}")
    print(f"👥 Participants: {participants}")
    print(f"🧪 Experiments: {list(experiments_to_run.keys())}")
    print(f"📊 Total experiments: {len(participants) * len(experiments_to_run)}")
    
    if args.dry_run:
        print("\n🔍 DRY RUN MODE - Commands will be printed but not executed")
    
    # Run experiments
    start_time = time.time()
    results = {}
    
    for participant in participants:
        results[participant] = {}
        print(f"\n👤 Processing participant: {participant}")
        
        for exp_key, exp_config in experiments_to_run.items():
            success = run_single_experiment(
                args.config, args.dataset_dir, participant, exp_config,
                study_dir, args.model, args.max_epochs, args.dry_run
            )
            results[participant][exp_key] = success
    
    # Summary
    total_time = time.time() - start_time
    total_experiments = len(participants) * len(experiments_to_run)
    successful_experiments = sum(sum(participant_results.values()) for participant_results in results.values())
    
    print(f"\n📊 EXPERIMENT SUITE SUMMARY")
    print(f"⏱️  Total time: {total_time/3600:.2f} hours")
    print(f"✅ Successful: {successful_experiments}/{total_experiments}")
    print(f"❌ Failed: {total_experiments - successful_experiments}/{total_experiments}")
    print(f"📁 Results saved in: {study_dir}")
    
    # Save results summary
    summary = {
        'execution_summary': {
            'total_time_seconds': total_time,
            'successful_experiments': successful_experiments,
            'failed_experiments': total_experiments - successful_experiments,
            'success_rate': successful_experiments / total_experiments
        },
        'results_by_participant': results
    }
    
    summary_file = study_dir / "execution_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    if not args.dry_run:
        print(f"\n🎯 Next steps:")
        print(f"1. Run analysis: python analyze_ablation_results.py {study_dir}")
        print(f"2. Generate figures: python generate_ablation_figures.py {study_dir}")
    
    return 0 if successful_experiments == total_experiments else 1

if __name__ == "__main__":
    sys.exit(main())