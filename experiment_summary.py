#!/usr/bin/env python3
"""
Compute average customization improvement across all experiments.
Shows F1 score improvement from base to target models on validation data.
"""

import os
import json
import pandas as pd
import numpy as np
import argparse
from typing import Dict, List
from pathlib import Path

def load_experiment_metrics(experiment_name: str) -> List[Dict]:
    """Load metrics from all runs in an experiment."""
    data = []
    experiment_dir = f'experiments/{experiment_name}'

    if not os.path.isdir(experiment_dir):
        print(f"Warning: Experiment directory '{experiment_dir}' not found")
        return data

    # Get all training run subdirectories
    training_runs = [d for d in os.listdir(experiment_dir)
                    if os.path.isdir(f'{experiment_dir}/{d}') and
                    os.path.exists(f'{experiment_dir}/{d}/metrics.json')]

    print(f"Loading {experiment_name}: found {len(training_runs)} runs")

    for run_id in training_runs:
        run_dir = f'{experiment_dir}/{run_id}'

        try:
            # Load hyperparameters for fold info
            hyperparams_file = f'{run_dir}/hyperparameters.json'
            if os.path.exists(hyperparams_file):
                with open(hyperparams_file, 'r') as f:
                    hyperparams = json.load(f)
                fold = hyperparams.get('fold', 'unknown')
            else:
                fold = 'unknown'

            # Load metrics
            with open(f'{run_dir}/metrics.json', 'r') as f:
                metrics = json.load(f)

            # Extract F1 scores
            base_f1 = metrics.get('best_target_val_f1_from_best_base_model')
            target_f1 = metrics.get('best_target_val_f1')

            if base_f1 is not None and target_f1 is not None:
                improvement = target_f1 - base_f1
                relative_improvement = (improvement / base_f1 * 100) if base_f1 > 0 else 0

                data.append({
                    'experiment': experiment_name,
                    'run_id': run_id,
                    'fold': fold,
                    'base_f1': base_f1,
                    'target_f1': target_f1,
                    'improvement': improvement,
                    'relative_improvement_pct': relative_improvement
                })
            else:
                print(f"  Warning: Missing F1 metrics in {run_id}")

        except Exception as e:
            print(f"  Error loading {run_id}: {e}")
            continue

    return data

def get_all_experiments() -> List[str]:
    """Get list of all experiment directories."""
    experiments_dir = 'experiments'
    if not os.path.exists(experiments_dir):
        return []

    experiments = []
    for item in os.listdir(experiments_dir):
        exp_path = os.path.join(experiments_dir, item)
        if os.path.isdir(exp_path):
            # Check if it has any run subdirectories with metrics
            run_dirs = [d for d in os.listdir(exp_path)
                       if os.path.isdir(os.path.join(exp_path, d)) and
                       os.path.exists(os.path.join(exp_path, d, 'metrics.json'))]
            if run_dirs:
                experiments.append(item)

    return sorted(experiments)

def compute_experiment_summary(experiment_data: List[Dict]) -> Dict:
    """Compute summary statistics for an experiment."""
    if not experiment_data:
        return {}

    df = pd.DataFrame(experiment_data)

    summary = {
        'experiment': experiment_data[0]['experiment'],
        'n_folds': len(df),
        'mean_base_f1': df['base_f1'].mean(),
        'std_base_f1': df['base_f1'].std(),
        'mean_target_f1': df['target_f1'].mean(),
        'std_target_f1': df['target_f1'].std(),
        'mean_improvement': df['improvement'].mean(),
        'std_improvement': df['improvement'].std(),
        'mean_relative_improvement': df['relative_improvement_pct'].mean(),
        'std_relative_improvement': df['relative_improvement_pct'].std(),
        'min_improvement': df['improvement'].min(),
        'max_improvement': df['improvement'].max(),
        'positive_improvements': (df['improvement'] > 0).sum(),
        'negative_improvements': (df['improvement'] < 0).sum(),
        'zero_improvements': (df['improvement'] == 0).sum()
    }

    return summary

def analyze_all_experiments(sort_by: str = 'mean_improvement', output_csv: bool = False):
    """Analyze customization improvement across all experiments."""

    print("🔍 EXPERIMENT F1 IMPROVEMENT ANALYSIS")
    print("="*60)

    # Get all experiments
    experiments = get_all_experiments()

    if not experiments:
        print("No experiments found in 'experiments/' directory")
        return

    print(f"Found {len(experiments)} experiments: {', '.join(experiments)}")
    print()

    # Load data for all experiments
    all_summaries = []
    all_data = []

    for exp_name in experiments:
        exp_data = load_experiment_metrics(exp_name)
        if exp_data:
            summary = compute_experiment_summary(exp_data)
            if summary:
                all_summaries.append(summary)
                all_data.extend(exp_data)

    if not all_summaries:
        print("No valid experiment data found")
        return

    # Convert to DataFrame for easy sorting and display
    summary_df = pd.DataFrame(all_summaries)

    # Sort by specified metric
    if sort_by in summary_df.columns:
        summary_df = summary_df.sort_values(sort_by, ascending=False)

    # Display results
    print(f"\n📊 EXPERIMENT SUMMARY (sorted by {sort_by})")
    print("-" * 120)

    header = f"{'Experiment':<20} {'Folds':<6} {'Base F1':<12} {'Target F1':<12} {'Improvement':<12} {'Rel. Imp. %':<12} {'Success Rate':<12}"
    print(header)
    print("-" * 120)

    for _, row in summary_df.iterrows():
        success_rate = f"{row['positive_improvements']}/{row['n_folds']}"

        line = (f"{row['experiment']:<20} "
                f"{row['n_folds']:<6} "
                f"{row['mean_base_f1']:.3f}±{row['std_base_f1']:.3f} "
                f"{row['mean_target_f1']:.3f}±{row['std_target_f1']:.3f} "
                f"{row['mean_improvement']:+.3f}±{row['std_improvement']:.3f} "
                f"{row['mean_relative_improvement']:+.1f}±{row['std_relative_improvement']:.1f}% "
                f"{success_rate:<12}")
        print(line)

    # Find best and worst by current sort metric
    best_exp = summary_df.iloc[0]
    worst_exp = summary_df.iloc[-1]

    # Find best by target F1 performance specifically
    best_f1_exp = summary_df.loc[summary_df['mean_target_f1'].idxmax()]

    print(f"\n🏆 BEST EXPERIMENT (by {sort_by}): {best_exp['experiment']}")
    print(f"   • Average improvement: {best_exp['mean_improvement']:+.3f} ± {best_exp['std_improvement']:.3f}")
    print(f"   • Relative improvement: {best_exp['mean_relative_improvement']:+.1f}% ± {best_exp['std_relative_improvement']:.1f}%")
    print(f"   • Target F1: {best_exp['mean_target_f1']:.3f} ± {best_exp['std_target_f1']:.3f}")
    print(f"   • Success rate: {best_exp['positive_improvements']}/{best_exp['n_folds']} folds")
    print(f"   • Range: {best_exp['min_improvement']:+.3f} to {best_exp['max_improvement']:+.3f}")

    if best_f1_exp['experiment'] != best_exp['experiment']:
        print(f"\n🎯 BEST TARGET F1 PERFORMANCE: {best_f1_exp['experiment']}")
        print(f"   • Target F1: {best_f1_exp['mean_target_f1']:.3f} ± {best_f1_exp['std_target_f1']:.3f}")
        print(f"   • Average improvement: {best_f1_exp['mean_improvement']:+.3f} ± {best_f1_exp['std_improvement']:.3f}")
        print(f"   • Relative improvement: {best_f1_exp['mean_relative_improvement']:+.1f}% ± {best_f1_exp['std_relative_improvement']:.1f}%")
        print(f"   • Success rate: {best_f1_exp['positive_improvements']}/{best_f1_exp['n_folds']} folds")

    print(f"\n🔴 WORST EXPERIMENT (by {sort_by}): {worst_exp['experiment']}")
    print(f"   • Average improvement: {worst_exp['mean_improvement']:+.3f} ± {worst_exp['std_improvement']:.3f}")
    print(f"   • Relative improvement: {worst_exp['mean_relative_improvement']:+.1f}% ± {worst_exp['std_relative_improvement']:.1f}%")
    print(f"   • Target F1: {worst_exp['mean_target_f1']:.3f} ± {worst_exp['std_target_f1']:.3f}")
    print(f"   • Success rate: {worst_exp['positive_improvements']}/{worst_exp['n_folds']} folds")
    print(f"   • Range: {worst_exp['min_improvement']:+.3f} to {worst_exp['max_improvement']:+.3f}")

    # Overall statistics
    all_data_df = pd.DataFrame(all_data)

    print(f"\n📈 OVERALL STATISTICS")
    print(f"   • Total runs analyzed: {len(all_data_df)}")
    print(f"   • Total experiments: {len(summary_df)}")
    print(f"   • Global mean improvement: {all_data_df['improvement'].mean():+.3f} ± {all_data_df['improvement'].std():.3f}")
    print(f"   • Global success rate: {(all_data_df['improvement'] > 0).sum()}/{len(all_data_df)} ({(all_data_df['improvement'] > 0).mean()*100:.1f}%)")

    # Statistical insights
    print(f"\n🔍 INSIGHTS")
    improvements = all_data_df['improvement']

    if improvements.mean() > 0:
        print(f"   • Overall positive trend: customization helps on average")
    else:
        print(f"   • Overall negative trend: customization hurts on average")

    # Check consistency
    consistent_experiments = summary_df[summary_df['positive_improvements'] == summary_df['n_folds']]
    if len(consistent_experiments) > 0:
        print(f"   • {len(consistent_experiments)} experiments show improvement on ALL folds:")
        for _, exp in consistent_experiments.iterrows():
            print(f"     - {exp['experiment']}: {exp['mean_improvement']:+.3f} improvement")

    # Check problematic experiments
    problematic_experiments = summary_df[summary_df['positive_improvements'] == 0]
    if len(problematic_experiments) > 0:
        print(f"   • {len(problematic_experiments)} experiments show NO improvement on any fold:")
        for _, exp in problematic_experiments.iterrows():
            print(f"     - {exp['experiment']}: {exp['mean_improvement']:+.3f} average")

    # Save CSV if requested
    if output_csv:
        csv_filename = 'experiment_improvement_summary.csv'
        summary_df.to_csv(csv_filename, index=False)
        print(f"\n💾 Summary saved to: {csv_filename}")

        # Also save detailed data
        detailed_filename = 'experiment_improvement_detailed.csv'
        all_data_df.to_csv(detailed_filename, index=False)
        print(f"💾 Detailed data saved to: {detailed_filename}")

def main():
    parser = argparse.ArgumentParser(description='Analyze F1 improvement across all experiments')
    parser.add_argument('--sort-by', default='mean_improvement',
                       choices=['mean_improvement', 'mean_relative_improvement', 'mean_target_f1', 'positive_improvements'],
                       help='Metric to sort experiments by (default: mean_improvement)')
    parser.add_argument('--csv', action='store_true',
                       help='Save results to CSV files')

    args = parser.parse_args()

    analyze_all_experiments(sort_by=args.sort_by, output_csv=args.csv)

if __name__ == "__main__":
    main()