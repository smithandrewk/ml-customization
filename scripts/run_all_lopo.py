#!/usr/bin/env python3
"""
Batch LOPO (Leave-One-Participant-Out) Experiments
==================================================

Systematically run customization training for all participants as targets.
This is the core experiment for the personalized health monitoring paper.

Usage:
    python scripts/run_all_lopo.py --dataset_dir data/001_participant_data --config configs/config.yaml

Output:
    - results/lopo/{participant}/ - Individual experiment results
    - results/lopo_summary.csv - Aggregated results for analysis
"""

import os
import sys
import argparse
import subprocess
import csv
import json
from datetime import datetime
from pathlib import Path
import torch

def get_available_participants(dataset_dir):
    """Get list of available participants from dataset directory."""
    participants = []
    if not os.path.exists(dataset_dir):
        raise ValueError(f"Dataset directory {dataset_dir} does not exist")
    
    for file in os.listdir(dataset_dir):
        if file.endswith('_train.pt'):
            participant = file.replace('_train.pt', '')
            # Check if corresponding test file exists
            test_file = f"{participant}_test.pt"
            if os.path.exists(os.path.join(dataset_dir, test_file)):
                participants.append(participant)
    
    return sorted(participants)

def run_customization_experiment(target_participant, dataset_dir, config_path, results_base_dir):
    """Run customization training for a single target participant."""
    print(f"\n{'='*60}")
    print(f"RUNNING LOPO EXPERIMENT: TARGET = {target_participant}")
    print(f"{'='*60}")
    
    # Create results directory for this participant
    participant_results_dir = os.path.join(results_base_dir, target_participant)
    os.makedirs(participant_results_dir, exist_ok=True)
    
    # Build command
    cmd = [
        'python3', 'train_customization.py',
        '--dataset_dir', dataset_dir,
        '--target_participant', target_participant,
        '--config', config_path
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Results will be saved to: {participant_results_dir}")
    
    # Run the experiment
    start_time = datetime.now()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            print(f"✅ SUCCESS: {target_participant} experiment completed")
            
            # Find the experiment directory (will be auto-generated)
            # Look for newest directory matching pattern
            experiment_dirs = []
            for d in os.listdir('experiments'):
                if f'custom_{target_participant}' in d and os.path.isdir(f'experiments/{d}'):
                    experiment_dirs.append(d)
            
            if experiment_dirs:
                # Get the most recent one
                latest_exp_dir = sorted(experiment_dirs)[-1]
                source_dir = f'experiments/{latest_exp_dir}'
                
                # Copy results to organized location
                import shutil
                if os.path.exists(source_dir):
                    # Copy key files
                    files_to_copy = [
                        'base_model.pt',
                        'customized_model.pt', 
                        'base_metrics.pt',
                        'custom_metrics.pt',
                        'results_summary.txt'
                    ]
                    
                    for file in files_to_copy:
                        src = os.path.join(source_dir, file)
                        dst = os.path.join(participant_results_dir, file)
                        if os.path.exists(src):
                            shutil.copy2(src, dst)
                    
                    # Copy plots
                    for plot_file in os.listdir(source_dir):
                        if plot_file.endswith(('.jpg', '.png', '.pdf')):
                            src = os.path.join(source_dir, plot_file)
                            dst = os.path.join(participant_results_dir, plot_file)
                            shutil.copy2(src, dst)
                    
                    print(f"📁 Results copied to: {participant_results_dir}")
                
                return {
                    'participant': target_participant,
                    'status': 'success',
                    'experiment_dir': source_dir,
                    'duration_minutes': (datetime.now() - start_time).total_seconds() / 60,
                    'stdout': result.stdout[-1000:],  # Last 1000 chars
                    'stderr': result.stderr[-1000:] if result.stderr else ''
                }
            else:
                print(f"⚠️  WARNING: Could not find experiment directory for {target_participant}")
                return {
                    'participant': target_participant,
                    'status': 'success_no_dir',
                    'duration_minutes': (datetime.now() - start_time).total_seconds() / 60,
                    'stdout': result.stdout[-1000:],
                    'stderr': result.stderr[-1000:] if result.stderr else ''
                }
        else:
            print(f"❌ FAILED: {target_participant} experiment failed")
            print(f"Return code: {result.returncode}")
            print(f"STDOUT: {result.stdout[-500:]}")  # Last 500 chars
            print(f"STDERR: {result.stderr[-500:]}")
            
            return {
                'participant': target_participant,
                'status': 'failed',
                'return_code': result.returncode,
                'duration_minutes': (datetime.now() - start_time).total_seconds() / 60,
                'stdout': result.stdout[-1000:],
                'stderr': result.stderr[-1000:]
            }
            
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT: {target_participant} experiment exceeded 1 hour limit")
        return {
            'participant': target_participant,
            'status': 'timeout',
            'duration_minutes': 60,
            'stdout': '',
            'stderr': 'Experiment timed out after 1 hour'
        }
    except Exception as e:
        print(f"💥 ERROR: {target_participant} experiment crashed: {str(e)}")
        return {
            'participant': target_participant,
            'status': 'error',
            'error': str(e),
            'duration_minutes': (datetime.now() - start_time).total_seconds() / 60,
            'stdout': '',
            'stderr': str(e)
        }

def extract_metrics_from_results(participant_results_dir):
    """Extract key metrics from experiment results."""
    try:
        # Load base and custom metrics
        base_metrics_path = os.path.join(participant_results_dir, 'base_metrics.pt')
        custom_metrics_path = os.path.join(participant_results_dir, 'custom_metrics.pt')
        
        if not (os.path.exists(base_metrics_path) and os.path.exists(custom_metrics_path)):
            return None
            
        base_metrics = torch.load(base_metrics_path, map_location='cpu')
        custom_metrics = torch.load(custom_metrics_path, map_location='cpu')
        
        # Extract key performance metrics
        return {
            'base_f1': base_metrics['best_f1'],
            'custom_f1': custom_metrics['best_target_f1'],
            'f1_improvement': custom_metrics['best_target_f1'] - base_metrics['best_f1'],
            'f1_improvement_percent': ((custom_metrics['best_target_f1'] - base_metrics['best_f1']) / base_metrics['best_f1']) * 100,
            'base_train_samples': base_metrics['base_train_samples'],
            'base_test_samples': base_metrics['base_test_samples'],
            'target_train_samples': custom_metrics['combined_train_samples'] - base_metrics['base_train_samples'],
            'target_test_samples': custom_metrics['target_test_samples'],
            'custom_learning_rate': custom_metrics['custom_learning_rate'],
            'base_epochs': len(base_metrics['train_f1']),
            'custom_epochs': len(custom_metrics['train_f1'])
        }
    except Exception as e:
        print(f"Warning: Could not extract metrics for participant: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Run LOPO experiments for all participants')
    parser.add_argument('--dataset_dir', type=str, required=True,
                       help='Directory containing participant-specific train/test files')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--results_dir', type=str, default='results/lopo',
                       help='Base directory for results')
    parser.add_argument('--participants', type=str, nargs='+',
                       help='Specific participants to run (default: all available)')
    parser.add_argument('--skip_existing', action='store_true',
                       help='Skip participants that already have results')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.dataset_dir):
        print(f"❌ Error: Dataset directory {args.dataset_dir} does not exist")
        sys.exit(1)
        
    if not os.path.exists(args.config):
        print(f"❌ Error: Config file {args.config} does not exist") 
        sys.exit(1)
    
    # Get participants
    if args.participants:
        participants = args.participants
        print(f"Running experiments for specified participants: {participants}")
    else:
        participants = get_available_participants(args.dataset_dir)
        print(f"Found {len(participants)} participants: {participants}")
    
    if not participants:
        print("❌ Error: No participants found")
        sys.exit(1)
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Run experiments
    print(f"\n🚀 STARTING BATCH LOPO EXPERIMENTS")
    print(f"📂 Dataset: {args.dataset_dir}")
    print(f"⚙️  Config: {args.config}")
    print(f"📊 Results: {args.results_dir}")
    print(f"👥 Participants: {len(participants)}")
    
    experiment_results = []
    successful_experiments = 0
    
    start_time = datetime.now()
    
    for i, participant in enumerate(participants, 1):
        print(f"\n🎯 EXPERIMENT {i}/{len(participants)}: {participant}")
        
        participant_results_dir = os.path.join(args.results_dir, participant)
        
        # Check if results already exist
        if args.skip_existing and os.path.exists(participant_results_dir):
            print(f"⏭️  SKIPPING: Results already exist for {participant}")
            continue
        
        # Run experiment
        result = run_customization_experiment(
            participant, args.dataset_dir, args.config, args.results_dir
        )
        experiment_results.append(result)
        
        if result['status'] == 'success':
            successful_experiments += 1
            
            # Extract performance metrics
            metrics = extract_metrics_from_results(participant_results_dir)
            if metrics:
                result.update(metrics)
                print(f"📈 Performance: Base F1={metrics['base_f1']:.4f} → Custom F1={metrics['custom_f1']:.4f} (+{metrics['f1_improvement']:+.4f})")
            
        # Save intermediate results
        results_file = os.path.join(args.results_dir, 'experiment_log.json')
        with open(results_file, 'w') as f:
            json.dump(experiment_results, f, indent=2, default=str)
    
    # Generate summary
    total_time = datetime.now() - start_time
    
    print(f"\n{'='*60}")
    print(f"BATCH LOPO EXPERIMENTS COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Successful: {successful_experiments}/{len(participants)}")
    print(f"⏱️  Total time: {total_time}")
    print(f"📊 Results saved to: {args.results_dir}")
    
    # Create summary CSV
    summary_file = os.path.join(args.results_dir, 'lopo_summary.csv')
    with open(summary_file, 'w', newline='') as f:
        if experiment_results:
            # Get all possible keys from results
            all_keys = set()
            for result in experiment_results:
                all_keys.update(result.keys())
            
            fieldnames = ['participant', 'status'] + sorted(all_keys - {'participant', 'status'})
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(experiment_results)
    
    print(f"📋 Summary CSV: {summary_file}")
    
    # Quick performance summary
    successful_results = [r for r in experiment_results if r.get('base_f1') is not None]
    if successful_results:
        improvements = [r['f1_improvement'] for r in successful_results]
        avg_improvement = sum(improvements) / len(improvements)
        positive_improvements = [imp for imp in improvements if imp > 0]
        
        print(f"\n📈 PERFORMANCE SUMMARY:")
        print(f"   • Average F1 improvement: {avg_improvement:+.4f}")
        print(f"   • Participants with positive improvement: {len(positive_improvements)}/{len(successful_results)}")
        if positive_improvements:
            print(f"   • Average positive improvement: {sum(positive_improvements)/len(positive_improvements):+.4f}")
    
    # Update research plan
    print(f"\n📝 Next steps:")
    print(f"   1. Review results in {summary_file}")
    print(f"   2. Run statistical analysis: python scripts/analyze_results.py")
    print(f"   3. Generate figures: python scripts/figure_generation/")
    
    if successful_experiments < len(participants):
        print(f"\n⚠️  Some experiments failed. Check experiment_log.json for details.")
        sys.exit(1)
    else:
        print(f"\n🎉 All experiments completed successfully!")

if __name__ == '__main__':
    main()