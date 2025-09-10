#!/usr/bin/env python3
"""
Ablation Study Results Analysis Script

This script analyzes the results from the experiment suite, computes statistics,
and prepares data for visualization.

Usage:
    python analyze_ablation_results.py /path/to/ablation_study_results
    python analyze_ablation_results.py /path/to/ablation_study_results --output_format csv
"""

import os
import sys
import argparse
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import glob
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Analyze ablation study results')
    parser.add_argument('study_dir', type=str, help='Path to ablation study results directory')
    parser.add_argument('--output_format', type=str, default='csv', choices=['csv', 'json', 'both'],
                       help='Output format for results')
    parser.add_argument('--significance_level', type=float, default=0.05,
                       help='Significance level for statistical tests')
    parser.add_argument('--baseline_experiment', type=str, default='baseline',
                       help='Name of baseline experiment for comparisons')
    
    return parser.parse_args()

def find_experiment_results(study_dir):
    """Find all experiment result files in the study directory."""
    study_path = Path(study_dir)
    
    # Look for experiment results in the experiments subdirectory
    experiments_dir = study_path / "experiments"
    if not experiments_dir.exists():
        print(f"❌ Experiments directory not found: {experiments_dir}")
        return {}
    
    results = {}
    
    # Find all experiment directories
    for exp_dir in experiments_dir.iterdir():
        if exp_dir.is_dir():
            # Parse experiment name: participant_technique
            dir_name = exp_dir.name
            if '_' in dir_name:
                parts = dir_name.split('_', 1)
                participant = parts[0]
                experiment = parts[1]
                
                # Look for custom_metrics.pt file
                metrics_file = exp_dir / "custom_metrics.pt"
                if metrics_file.exists():
                    if participant not in results:
                        results[participant] = {}
                    results[participant][experiment] = str(metrics_file)
                else:
                    print(f"⚠️  Missing metrics file: {metrics_file}")
    
    return results

def load_experiment_metrics(metrics_file_path):
    """Load metrics from a single experiment."""
    try:
        metrics = torch.load(metrics_file_path, map_location='cpu')
        
        # Extract key performance metrics
        result = {
            # Core performance metrics
            'base_model_test_f1': metrics.get('base_model_test_f1', 0.0),
            'custom_model_test_f1': metrics.get('custom_model_test_f1', 0.0),
            'absolute_improvement': metrics.get('absolute_improvement', 0.0),
            'percentage_improvement': metrics.get('percentage_improvement', 0.0),
            
            # Training details
            'base_best_metric': metrics.get('base_best_metric', 0.0),
            'custom_best_metric': metrics.get('custom_best_metric', 0.0),
            'base_best_epoch': metrics.get('base_best_epoch', 0),
            'custom_best_epoch': metrics.get('custom_best_epoch', 0),
            
            # Advanced techniques used
            'use_layerwise_finetuning': metrics.get('use_layerwise_finetuning', False),
            'use_gradual_unfreezing': metrics.get('use_gradual_unfreezing', False),
            'use_ewc': metrics.get('use_ewc', False),
            'use_augmentation': metrics.get('use_augmentation', False),
            'use_coral': metrics.get('use_coral', False),
            'use_ensemble': metrics.get('use_ensemble', False),
            'use_contrastive': metrics.get('use_contrastive', False),
            
            # Technique parameters
            'ewc_lambda': metrics.get('ewc_lambda', 0),
            'coral_lambda': metrics.get('coral_lambda', 0),
            'contrastive_lambda': metrics.get('contrastive_lambda', 0),
            
            # Data information
            'target_participant': metrics.get('target_participant', ''),
            'combined_train_samples': metrics.get('combined_train_samples', 0),
            'target_val_samples': metrics.get('target_val_samples', 0),
            'target_test_samples': metrics.get('target_test_samples', 0),
            
            # Ensemble results (if applicable)
            'ensemble_test_f1': metrics.get('ensemble_test_f1', None),
            'ensemble_alpha': metrics.get('ensemble_alpha', None),
            
            # Meta information
            'evaluation_method': metrics.get('evaluation_method', 'moving_average'),
            'early_stopping_metric': metrics.get('early_stopping_metric', 'f1'),
            'ma_window_size': metrics.get('ma_window_size', 5)
        }
        
        return result
        
    except Exception as e:
        print(f"❌ Error loading {metrics_file_path}: {e}")
        return None

def compute_summary_statistics(df, metric_col='absolute_improvement'):
    """Compute summary statistics for a metric across experiments."""
    stats_dict = {
        'count': len(df),
        'mean': df[metric_col].mean(),
        'std': df[metric_col].std(),
        'min': df[metric_col].min(),
        'max': df[metric_col].max(),
        'median': df[metric_col].median(),
        'q25': df[metric_col].quantile(0.25),
        'q75': df[metric_col].quantile(0.75),
        'sem': df[metric_col].sem(),  # Standard error of mean
    }
    
    # Add confidence interval (95% by default)
    ci_95 = stats.t.interval(0.95, len(df)-1, loc=stats_dict['mean'], scale=stats_dict['sem'])
    stats_dict['ci_95_lower'] = ci_95[0]
    stats_dict['ci_95_upper'] = ci_95[1]
    
    return stats_dict

def perform_statistical_tests(df, baseline_experiment, significance_level=0.05):
    """Perform statistical tests comparing each technique to baseline."""
    if baseline_experiment not in df['experiment'].values:
        print(f"⚠️  Baseline experiment '{baseline_experiment}' not found in results")
        return {}
    
    baseline_data = df[df['experiment'] == baseline_experiment]['absolute_improvement']
    
    test_results = {}
    
    for experiment in df['experiment'].unique():
        if experiment == baseline_experiment:
            continue
            
        experiment_data = df[df['experiment'] == experiment]['absolute_improvement']
        
        if len(experiment_data) < 2 or len(baseline_data) < 2:
            test_results[experiment] = {
                'test': 'insufficient_data',
                'p_value': None,
                'significant': False,
                'effect_size': None
            }
            continue
        
        # Perform paired t-test (assuming same participants)
        if len(experiment_data) == len(baseline_data):
            try:
                t_stat, p_value = stats.ttest_rel(experiment_data, baseline_data)
                test_type = 'paired_t_test'
            except:
                t_stat, p_value = stats.ttest_ind(experiment_data, baseline_data)
                test_type = 'independent_t_test'
        else:
            # Independent t-test if different sample sizes
            t_stat, p_value = stats.ttest_ind(experiment_data, baseline_data)
            test_type = 'independent_t_test'
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(experiment_data) - 1) * experiment_data.std()**2 + 
                             (len(baseline_data) - 1) * baseline_data.std()**2) / 
                            (len(experiment_data) + len(baseline_data) - 2))
        
        cohens_d = (experiment_data.mean() - baseline_data.mean()) / pooled_std if pooled_std > 0 else 0
        
        test_results[experiment] = {
            'test': test_type,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < significance_level,
            'effect_size_cohens_d': cohens_d,
            'mean_difference': experiment_data.mean() - baseline_data.mean()
        }
    
    return test_results

def create_technique_analysis(df):
    """Analyze the impact of individual techniques."""
    # Create binary columns for each technique
    technique_cols = [
        'use_layerwise_finetuning', 'use_gradual_unfreezing', 'use_ewc',
        'use_augmentation', 'use_coral', 'use_ensemble', 'use_contrastive'
    ]
    
    technique_analysis = {}
    
    for technique in technique_cols:
        if technique in df.columns:
            # Compare experiments with and without this technique
            with_technique = df[df[technique] == True]['absolute_improvement']
            without_technique = df[df[technique] == False]['absolute_improvement']
            
            if len(with_technique) > 0 and len(without_technique) > 0:
                # Statistical test
                t_stat, p_value = stats.ttest_ind(with_technique, without_technique)
                
                technique_analysis[technique] = {
                    'with_technique': {
                        'count': len(with_technique),
                        'mean': with_technique.mean(),
                        'std': with_technique.std()
                    },
                    'without_technique': {
                        'count': len(without_technique),
                        'mean': without_technique.mean(),
                        'std': without_technique.std()
                    },
                    'mean_difference': with_technique.mean() - without_technique.mean(),
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
    
    return technique_analysis

def analyze_participant_variation(df):
    """Analyze how much performance varies across participants."""
    participant_stats = df.groupby('participant')['absolute_improvement'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).reset_index()
    
    # Overall variation across participants
    overall_stats = {
        'mean_of_means': participant_stats['mean'].mean(),
        'std_of_means': participant_stats['mean'].std(),
        'min_participant_mean': participant_stats['mean'].min(),
        'max_participant_mean': participant_stats['mean'].max(),
        'participant_range': participant_stats['mean'].max() - participant_stats['mean'].min()
    }
    
    return participant_stats, overall_stats

def create_ranking_analysis(df):
    """Create ranking of techniques by performance."""
    # Group by experiment and compute statistics
    experiment_stats = df.groupby('experiment').agg({
        'absolute_improvement': ['count', 'mean', 'std', 'sem'],
        'percentage_improvement': ['mean', 'std'],
        'custom_model_test_f1': ['mean', 'std']
    }).round(4)
    
    # Flatten column names
    experiment_stats.columns = [f"{col[1]}_{col[0]}" if col[1] else col[0] 
                               for col in experiment_stats.columns]
    
    # Sort by mean absolute improvement
    experiment_stats = experiment_stats.sort_values('mean_absolute_improvement', ascending=False)
    
    # Add rank
    experiment_stats['rank'] = range(1, len(experiment_stats) + 1)
    
    return experiment_stats

def main():
    args = parse_arguments()
    
    study_dir = Path(args.study_dir)
    if not study_dir.exists():
        print(f"❌ Study directory not found: {study_dir}")
        sys.exit(1)
    
    print(f"📊 Analyzing ablation study results: {study_dir}")
    
    # Find all experiment results
    experiment_files = find_experiment_results(study_dir)
    
    if not experiment_files:
        print("❌ No experiment results found")
        sys.exit(1)
    
    print(f"📁 Found results for {len(experiment_files)} participants")
    
    # Load all results into a DataFrame
    all_results = []
    
    for participant, experiments in experiment_files.items():
        print(f"📈 Processing participant: {participant}")
        
        for experiment, metrics_file in experiments.items():
            metrics = load_experiment_metrics(metrics_file)
            
            if metrics:
                metrics['participant'] = participant
                metrics['experiment'] = experiment
                metrics['metrics_file'] = metrics_file
                all_results.append(metrics)
                print(f"   ✅ {experiment}: F1 improvement = {metrics['absolute_improvement']:.4f}")
            else:
                print(f"   ❌ Failed to load: {experiment}")
    
    if not all_results:
        print("❌ No valid results loaded")
        sys.exit(1)
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    print(f"\n📊 ANALYSIS SUMMARY")
    print(f"Total experiments: {len(df)}")
    print(f"Participants: {df['participant'].nunique()}")
    print(f"Techniques tested: {df['experiment'].nunique()}")
    
    # Compute overall statistics
    overall_stats = compute_summary_statistics(df)
    print(f"\nOverall absolute improvement: {overall_stats['mean']:.4f} ± {overall_stats['std']:.4f}")
    
    # Perform statistical tests
    statistical_tests = perform_statistical_tests(df, args.baseline_experiment)
    
    # Individual technique analysis
    technique_analysis = create_technique_analysis(df)
    
    # Participant variation analysis
    participant_stats, participant_variation = analyze_participant_variation(df)
    
    # Ranking analysis
    ranking_analysis = create_ranking_analysis(df)
    
    # Create comprehensive analysis report
    analysis_results = {
        'metadata': {
            'study_directory': str(study_dir),
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'total_experiments': len(df),
            'participants': sorted(df['participant'].unique().tolist()),
            'techniques': sorted(df['experiment'].unique().tolist())
        },
        'overall_statistics': overall_stats,
        'statistical_tests': statistical_tests,
        'technique_analysis': technique_analysis,
        'participant_variation': participant_variation,
        'ranking': ranking_analysis.to_dict('index')
    }
    
    # Save results
    analysis_dir = study_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    
    if args.output_format in ['csv', 'both']:
        # Save main results as CSV
        csv_file = analysis_dir / "ablation_results.csv"
        df.to_csv(csv_file, index=False)
        print(f"💾 Results saved: {csv_file}")
        
        # Save ranking as CSV
        ranking_csv = analysis_dir / "technique_ranking.csv"
        ranking_analysis.to_csv(ranking_csv)
        print(f"💾 Ranking saved: {ranking_csv}")
        
        # Save participant stats as CSV
        participant_csv = analysis_dir / "participant_stats.csv"
        participant_stats.to_csv(participant_csv, index=False)
        print(f"💾 Participant stats saved: {participant_csv}")
    
    if args.output_format in ['json', 'both']:
        # Save comprehensive analysis as JSON
        json_file = analysis_dir / "comprehensive_analysis.json"
        with open(json_file, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        print(f"💾 Comprehensive analysis saved: {json_file}")
    
    # Print summary
    print(f"\n🏆 TOP TECHNIQUES (by mean absolute improvement):")
    for i, (experiment, data) in enumerate(ranking_analysis.head(5).iterrows(), 1):
        improvement = data['mean_absolute_improvement']
        std_err = data['sem_absolute_improvement']
        print(f"{i:2d}. {experiment:20s}: {improvement:+.4f} ± {std_err:.4f}")
    
    # Print significant improvements
    print(f"\n📈 SIGNIFICANT IMPROVEMENTS vs {args.baseline_experiment}:")
    significant_techniques = [(exp, data) for exp, data in statistical_tests.items() 
                             if data.get('significant', False)]
    
    if significant_techniques:
        for exp, data in sorted(significant_techniques, key=lambda x: x[1]['mean_difference'], reverse=True):
            improvement = data['mean_difference']
            p_value = data['p_value']
            effect_size = data['effect_size_cohens_d']
            print(f"   • {exp:20s}: {improvement:+.4f} (p={p_value:.4f}, d={effect_size:.3f})")
    else:
        print("   None found at α = 0.05 level")
    
    print(f"\n🎯 Next step: python generate_ablation_figures.py {study_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())