#!/usr/bin/env python3
"""
Robust comparison of ML experiments with leave-one-out cross-validation.
Compares paired fold results across different experimental conditions.
"""

import os
import json
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from typing import Dict, List, Tuple

def load_experiment_data(experiment_name: str) -> List[Dict]:
    """Load all training runs from an experiment directory."""
    data = []
    experiment_dir = f'experiments/{experiment_name}'

    if not os.path.isdir(experiment_dir):
        print(f"Warning: Experiment directory '{experiment_dir}' not found")
        return data

    # Get all training run subdirectories
    training_runs = [d for d in os.listdir(experiment_dir)
                    if os.path.isdir(f'{experiment_dir}/{d}') and
                    os.path.exists(f'{experiment_dir}/{d}/hyperparameters.json')]

    print(f"Loading {experiment_name}: found {len(training_runs)} runs")

    for run_id in training_runs:
        run_dir = f'{experiment_dir}/{run_id}'

        # Check all required files exist
        required_files = ['hyperparameters.json', 'metrics.json', 'losses.json']
        if not all(os.path.exists(f'{run_dir}/{file}') for file in required_files):
            print(f"  Skipping {run_id}: missing required files")
            continue

        try:
            with open(f'{run_dir}/hyperparameters.json', 'r') as f:
                hyperparameters = json.load(f)

            with open(f'{run_dir}/metrics.json', 'r') as f:
                metrics = json.load(f)

            fold = hyperparameters['fold']
            participants = ['alsaad','anam','asfik','ejaz','iftakhar','tonmoy','unk1','dennis']
            target_participant = participants[fold] if fold < len(participants) else f'fold_{fold}'

            base_f1 = metrics['best_target_val_f1_from_best_base_model']
            target_f1 = metrics['best_target_val_f1']
            improvement = target_f1 - base_f1

            data.append({
                'experiment': experiment_name,
                'run_id': run_id,
                'fold': fold,
                'participant': target_participant,
                'base_f1': base_f1,
                'target_f1': target_f1,
                'absolute_improvement': improvement,
                'relative_improvement_pct': improvement / base_f1 * 100,
                'inverse_relative_improvement': (target_f1 - base_f1) / (1 - base_f1) * 100,
                'transition_epoch': metrics.get('transition_epoch', 0)
            })

        except Exception as e:
            print(f"  Error loading {run_id}: {e}")
            continue

    return data

def compute_comparison_stats(exp1_data: List[Dict], exp2_data: List[Dict],
                           metric: str) -> Dict:
    """Compute statistical comparison between two experiments for a given metric."""

    # Create DataFrames for easier manipulation
    df1 = pd.DataFrame(exp1_data)
    df2 = pd.DataFrame(exp2_data)

    # Ensure we have matching folds
    common_folds = set(df1['fold']) & set(df2['fold'])
    if len(common_folds) == 0:
        return {'error': 'No matching folds found'}

    # Sort by fold to ensure proper pairing
    df1_matched = df1[df1['fold'].isin(common_folds)].sort_values('fold')
    df2_matched = df2[df2['fold'].isin(common_folds)].sort_values('fold')

    if len(df1_matched) != len(df2_matched):
        return {'error': f'Unequal number of matching folds: {len(df1_matched)} vs {len(df2_matched)}'}

    values1 = df1_matched[metric].values
    values2 = df2_matched[metric].values

    # Basic statistics
    mean1, std1 = np.mean(values1), np.std(values1, ddof=1)
    mean2, std2 = np.mean(values2), np.std(values2, ddof=1)

    # Paired statistical tests
    t_stat, p_val_paired = stats.ttest_rel(values2, values1)  # exp2 vs exp1
    try:
        w_stat, p_val_wilcoxon = stats.wilcoxon(values2, values1, alternative='two-sided')
    except ValueError:
        # All differences are zero
        w_stat, p_val_wilcoxon = 0, 1.0

    # Effect size (Cohen's d for paired samples)
    differences = values2 - values1
    cohens_d = np.mean(differences) / np.std(differences, ddof=1) if np.std(differences, ddof=1) > 0 else 0

    # Unpaired tests for comparison
    t_stat_unpaired, p_val_unpaired = stats.ttest_ind(values2, values1)

    return {
        'n_pairs': len(values1),
        'exp1_mean': mean1,
        'exp1_std': std1,
        'exp2_mean': mean2,
        'exp2_std': std2,
        'mean_difference': mean2 - mean1,
        'cohens_d': cohens_d,
        'paired_t_stat': t_stat,
        'paired_p_value': p_val_paired,
        'wilcoxon_stat': w_stat,
        'wilcoxon_p_value': p_val_wilcoxon,
        'unpaired_t_stat': t_stat_unpaired,
        'unpaired_p_value': p_val_unpaired,
        'values1': values1,
        'values2': values2,
        'differences': differences
    }

def format_p_value(p: float) -> str:
    """Format p-value for display."""
    if p < 0.001:
        return "P < 0.001"
    elif p < 0.01:
        return f"P < 0.01"
    elif p < 0.05:
        return f"P < 0.05"
    else:
        return f"P = {p:.3f}"

def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"

def compare_experiments(exp1_name: str, exp2_name: str, output_dir: str = "figures",
                       include_training_analysis: bool = False, analyze_best_runs: bool = False):
    """Comprehensively compare two experiments."""

    print(f"\n{'='*60}")
    print(f"COMPARING EXPERIMENTS: {exp1_name} vs {exp2_name}")
    print(f"{'='*60}")

    # Load data
    exp1_data = load_experiment_data(exp1_name)
    exp2_data = load_experiment_data(exp2_name)

    if not exp1_data or not exp2_data:
        print("Error: Could not load data from both experiments")
        return

    # Metrics to compare
    metrics = [
        ('base_f1', 'Base F1 Score'),
        ('target_f1', 'Customized F1 Score'),
        ('absolute_improvement', 'Absolute Improvement (ΔF1)'),
        ('relative_improvement_pct', 'Relative Improvement (%)'),
        ('inverse_relative_improvement', 'Inverse Relative Improvement (%)')
    ]

    # Create comparison table
    results = []

    print(f"\n📊 STATISTICAL COMPARISON SUMMARY")
    exp1_header = f"{exp1_name} Mean±SD"
    exp2_header = f"{exp2_name} Mean±SD"
    print(f"{'Metric':<28} {exp1_header:<18} {exp2_header:<18} {'Difference':<12} {'Effect Size':<15} {'Paired P':<12} {'Unpaired P':<12} {'Significance'}")
    print("-" * 140)

    for metric_key, metric_name in metrics:
        stats_result = compute_comparison_stats(exp1_data, exp2_data, metric_key)

        if 'error' in stats_result:
            print(f"{metric_name:<25} ERROR: {stats_result['error']}")
            continue

        # Format results for display
        exp1_mean_str = f"{stats_result['exp1_mean']:.3f}±{stats_result['exp1_std']:.3f}"
        exp2_mean_str = f"{stats_result['exp2_mean']:.3f}±{stats_result['exp2_std']:.3f}"
        diff_str = f"{stats_result['mean_difference']:+.3f}"
        effect_str = f"{stats_result['cohens_d']:.2f} ({interpret_effect_size(stats_result['cohens_d'])})"
        paired_p_str = format_p_value(stats_result['paired_p_value'])
        unpaired_p_str = format_p_value(stats_result['unpaired_p_value'])

        # Determine significance (using paired test as primary)
        if stats_result['paired_p_value'] < 0.001:
            sig = "***"
        elif stats_result['paired_p_value'] < 0.01:
            sig = "**"
        elif stats_result['paired_p_value'] < 0.05:
            sig = "*"
        else:
            sig = "ns"

        # Add note if unpaired is significant but paired isn't
        if stats_result['paired_p_value'] >= 0.05 and stats_result['unpaired_p_value'] < 0.05:
            sig += "(u*)"  # unpaired significant

        print(f"{metric_name:<28} {exp1_mean_str:<18} {exp2_mean_str:<18} {diff_str:<12} {effect_str:<15} {paired_p_str:<12} {unpaired_p_str:<12} {sig}")

        results.append({
            'metric': metric_name,
            'metric_key': metric_key,
            **stats_result
        })

    # Detailed statistical breakdown
    print(f"\n🔍 DETAILED STATISTICAL ANALYSIS")
    print(f"Sample size: {results[0]['n_pairs']} paired comparisons")
    print(f"Tests performed:")
    print(f"  • Paired t-test: Compares fold-by-fold (recommended for your CV design)")
    print(f"  • Unpaired t-test: Compares overall distributions (shows aggregate differences)")
    print(f"  • Wilcoxon signed-rank: Non-parametric alternative to paired t-test")
    print(f"Effect size: Cohen's d for paired samples")
    print(f"Significance levels: *** P<0.001, ** P<0.01, * P<0.05, ns P≥0.05, (u*) unpaired-only significant")

    # Create visualization
    create_comparison_plots(exp1_data, exp2_data, exp1_name, exp2_name, results, output_dir)

    # Overall recommendation
    print(f"\n🎯 COMPREHENSIVE RECOMMENDATIONS")

    # Analyze all metrics for significant differences
    significant_results = []

    print(f"\n📈 SIGNIFICANT DIFFERENCES FOUND:")
    for result in results:
        if result['paired_p_value'] < 0.05:
            # Determine which experiment is actually higher
            if result['exp2_mean'] > result['exp1_mean']:
                winner = exp2_name
                direction = "higher"
            else:
                winner = exp1_name
                direction = "higher"

            effect_magnitude = interpret_effect_size(result['cohens_d'])

            print(f"• {result['metric']}: {winner} significantly {direction} ({effect_magnitude} effect, P={result['paired_p_value']:.3f})")
            significant_results.append({
                'metric': result['metric_key'],
                'winner': winner,
                'effect_size': abs(result['cohens_d']),
                'p_value': result['paired_p_value']
            })
        elif result['unpaired_p_value'] < 0.05:
            # Determine which experiment is actually higher
            if result['exp2_mean'] > result['exp1_mean']:
                winner = exp2_name
                direction = "higher"
            else:
                winner = exp1_name
                direction = "higher"
            print(f"• {result['metric']}: {winner} {direction} in overall distribution (unpaired test only, P={result['unpaired_p_value']:.3f})")

    if not significant_results:
        print("• No statistically significant differences detected")

    # Priority-based analysis
    print(f"\n🎯 PRIORITY-BASED ANALYSIS:")

    # Define metric priorities and check results
    metric_priorities = {
        'target_f1': {'weight': 5, 'description': 'Final Performance (most important)'},
        'inverse_relative_improvement': {'weight': 4, 'description': 'Learning Efficiency (very important)'},
        'absolute_improvement': {'weight': 3, 'description': 'Raw Performance Gain (important)'},
        'base_f1': {'weight': 2, 'description': 'Baseline Consistency (moderately important)'},
        'relative_improvement_pct': {'weight': 1, 'description': 'Relative Gain (least important)'}
    }

    total_score_exp1 = 0
    total_score_exp2 = 0
    total_possible = sum(p['weight'] for p in metric_priorities.values())

    for result in results:
        metric_key = result['metric_key']
        if metric_key in metric_priorities:
            weight = metric_priorities[metric_key]['weight']
            description = metric_priorities[metric_key]['description']

            if result['paired_p_value'] < 0.05:
                if result['mean_difference'] > 0:
                    total_score_exp2 += weight
                    print(f"• {description}: {exp2_name} wins (+{weight} points)")
                else:
                    total_score_exp1 += weight
                    print(f"• {description}: {exp1_name} wins (+{weight} points)")
            else:
                print(f"• {description}: No significant difference (0 points)")

    print(f"\n📊 WEIGHTED SCORES:")
    print(f"• {exp1_name}: {total_score_exp1}/{total_possible} points")
    print(f"• {exp2_name}: {total_score_exp2}/{total_possible} points")

    # Final recommendation
    print(f"\n🏆 FINAL RECOMMENDATION:")

    if total_score_exp2 > total_score_exp1:
        score_diff = total_score_exp2 - total_score_exp1
        if score_diff >= 7:  # High confidence threshold
            print(f"Strong recommendation: Use {exp2_name}")
            print(f"• Wins on key metrics with substantial advantage ({score_diff} point difference)")
        else:
            print(f"Moderate recommendation: Use {exp2_name}")
            print(f"• Slight advantage on important metrics ({score_diff} point difference)")
    elif total_score_exp1 > total_score_exp2:
        score_diff = total_score_exp1 - total_score_exp2
        if score_diff >= 7:
            print(f"Strong recommendation: Use {exp1_name}")
            print(f"• Wins on key metrics with substantial advantage ({score_diff} point difference)")
        else:
            print(f"Moderate recommendation: Use {exp1_name}")
            print(f"• Slight advantage on important metrics ({score_diff} point difference)")
    else:
        print(f"No clear recommendation: Methods are equivalent")
        print(f"• Consider other factors (computational cost, complexity, interpretability)")

    # Additional considerations
    if significant_results:
        largest_effect = max(significant_results, key=lambda x: x['effect_size'])
        print(f"\n💪 LARGEST EFFECT: {largest_effect['metric'].replace('_', ' ').title()} (Cohen's d = {largest_effect['effect_size']:.2f})")

        most_significant = min(significant_results, key=lambda x: x['p_value'])
        print(f"🎯 MOST SIGNIFICANT: {most_significant['metric'].replace('_', ' ').title()} (P = {most_significant['p_value']:.3f})")

    # Check for cases where unpaired tests show different results
    print(f"\n📋 PAIRED vs UNPAIRED TEST INTERPRETATION:")
    key_metrics_for_interpretation = ['target_f1', 'inverse_relative_improvement']
    for result in results:
        if result['metric_key'] in key_metrics_for_interpretation:
            paired_sig = result['paired_p_value'] < 0.05
            unpaired_sig = result['unpaired_p_value'] < 0.05

            if paired_sig and unpaired_sig:
                print(f"✅ {result['metric']}: Both tests agree - significant difference")
            elif not paired_sig and not unpaired_sig:
                print(f"❌ {result['metric']}: Both tests agree - no significant difference")
            elif not paired_sig and unpaired_sig:
                print(f"⚠️  {result['metric']}: Only unpaired significant - suggests overall difference but high participant variance")
            elif paired_sig and not unpaired_sig:
                print(f"🎯 {result['metric']}: Only paired significant - true experimental effect despite similar overall distributions")

    print(f"\n💡 INTERPRETATION GUIDE:")
    print(f"• Paired tests are more appropriate for your cross-validation design")
    print(f"• Unpaired tests show what you see visually in the box plots")
    print(f"• When only unpaired is significant: methods differ overall but participant variance is high")
    print(f"• When only paired is significant: true experimental effect with consistent fold-wise improvements")

    # Optional deep training analysis
    if include_training_analysis or analyze_best_runs:
        print(f"\n🔬 DEEP TRAINING DYNAMICS ANALYSIS")

        try:
            # Import training analyzer
            import sys
            sys.path.append('.')
            from analyze_training import analyze_single_run, TrainingAnalyzer

            if analyze_best_runs:
                # Find best performing runs from each experiment
                print(f"\n🏆 ANALYZING BEST PERFORMING RUNS:")

                # Find best run from each experiment based on target F1
                best_exp1_run = max(exp1_data, key=lambda x: x['target_f1'])
                best_exp2_run = max(exp2_data, key=lambda x: x['target_f1'])

                print(f"\n📊 Best {exp1_name} run: {best_exp1_run['run_id']} (F1: {best_exp1_run['target_f1']:.3f})")
                exp1_analysis = analyze_single_run(f"experiments/{exp1_name}", best_exp1_run['run_id'], save_results=True)

                print(f"\n📊 Best {exp2_name} run: {best_exp2_run['run_id']} (F1: {best_exp2_run['target_f1']:.3f})")
                exp2_analysis = analyze_single_run(f"experiments/{exp2_name}", best_exp2_run['run_id'], save_results=True)

                # Compare training health scores
                exp1_health = exp1_analysis.get('overall_health_score', 0)
                exp2_health = exp2_analysis.get('overall_health_score', 0)

                print(f"\n🏥 TRAINING HEALTH COMPARISON:")
                print(f"• {exp1_name} best run health: {exp1_health:.1f}/100")
                print(f"• {exp2_name} best run health: {exp2_health:.1f}/100")

                if exp1_health > exp2_health + 10:
                    print(f"• {exp1_name} has significantly healthier training dynamics")
                elif exp2_health > exp1_health + 10:
                    print(f"• {exp2_name} has significantly healthier training dynamics")
                else:
                    print(f"• Both experiments have similar training health")

                # Generate training dashboards for best runs
                print(f"\n📈 Generating training dashboards for best runs...")

                analyzer1 = TrainingAnalyzer(f"experiments/{exp1_name}", best_exp1_run['run_id'])
                analyzer1.training_health = exp1_analysis
                dashboard1_path = analyzer1.create_training_dashboard()
                print(f"• {exp1_name} dashboard: {dashboard1_path}")

                analyzer2 = TrainingAnalyzer(f"experiments/{exp2_name}", best_exp2_run['run_id'])
                analyzer2.training_health = exp2_analysis
                dashboard2_path = analyzer2.create_training_dashboard()
                print(f"• {exp2_name} dashboard: {dashboard2_path}")

            elif include_training_analysis:
                print(f"\n📊 TRAINING HEALTH SUMMARY ACROSS ALL RUNS:")

                # Analyze all runs and compute average health scores
                exp1_healths = []
                exp2_healths = []

                print(f"\n{exp1_name} runs:")
                for run_data in exp1_data[:3]:  # Limit to first 3 runs for brevity
                    try:
                        analysis = analyze_single_run(f"experiments/{exp1_name}", run_data['run_id'], save_results=False)
                        health_score = analysis.get('overall_health_score', 0)
                        exp1_healths.append(health_score)
                        print(f"• Run {run_data['run_id']}: Health {health_score:.1f}/100")
                    except Exception as e:
                        print(f"• Run {run_data['run_id']}: Analysis failed ({str(e)[:50]}...)")

                print(f"\n{exp2_name} runs:")
                for run_data in exp2_data[:3]:  # Limit to first 3 runs for brevity
                    try:
                        analysis = analyze_single_run(f"experiments/{exp2_name}", run_data['run_id'], save_results=False)
                        health_score = analysis.get('overall_health_score', 0)
                        exp2_healths.append(health_score)
                        print(f"• Run {run_data['run_id']}: Health {health_score:.1f}/100")
                    except Exception as e:
                        print(f"• Run {run_data['run_id']}: Analysis failed ({str(e)[:50]}...)")

                # Compare average health scores
                if exp1_healths and exp2_healths:
                    avg_health1 = np.mean(exp1_healths)
                    avg_health2 = np.mean(exp2_healths)

                    print(f"\n🏥 AVERAGE TRAINING HEALTH:")
                    print(f"• {exp1_name}: {avg_health1:.1f}/100 (±{np.std(exp1_healths):.1f})")
                    print(f"• {exp2_name}: {avg_health2:.1f}/100 (±{np.std(exp2_healths):.1f})")

                    if avg_health1 > avg_health2 + 5:
                        print(f"• {exp1_name} has better overall training health")
                    elif avg_health2 > avg_health1 + 5:
                        print(f"• {exp2_name} has better overall training health")
                    else:
                        print(f"• Both experiments have similar training health")

        except ImportError:
            print(f"⚠️  Training analysis module not available. Run 'pip install pandas' if needed.")
        except Exception as e:
            print(f"⚠️  Training analysis failed: {str(e)[:100]}...")
            print(f"💡 Try running: python3 analyze_training.py {exp1_name} <run_id> --visualize")

def create_comparison_plots(exp1_data: List[Dict], exp2_data: List[Dict],
                          exp1_name: str, exp2_name: str, results: List[Dict],
                          output_dir: str):
    """Create comprehensive comparison visualizations."""

    # Set up the plotting style
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14
    })

    # Create figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(f'Experiment Comparison: {exp1_name} vs {exp2_name}', fontsize=16, fontweight='bold')

    # Colors
    colors = {'exp1': '#1f77b4', 'exp2': '#d62728', 'improve': '#2ca02c', 'neutral': '#7f7f7f'}

    # Convert to DataFrames
    df1 = pd.DataFrame(exp1_data)
    df2 = pd.DataFrame(exp2_data)

    # Ensure matching folds
    common_folds = set(df1['fold']) & set(df2['fold'])
    df1 = df1[df1['fold'].isin(common_folds)].sort_values('fold')
    df2 = df2[df2['fold'].isin(common_folds)].sort_values('fold')

    # Plot 1: Base F1 comparison
    ax1 = axes[0, 0]
    x = np.arange(len(df1))
    width = 0.35
    ax1.bar(x - width/2, df1['base_f1'], width, label=exp1_name, color=colors['exp1'], alpha=0.7)
    ax1.bar(x + width/2, df2['base_f1'], width, label=exp2_name, color=colors['exp2'], alpha=0.7)
    ax1.set_xlabel('Fold (Participant)')
    ax1.set_ylabel('Base F1 Score')
    ax1.set_title('Base Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'P{f}' for f in df1['fold']], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Target F1 comparison
    ax2 = axes[0, 1]
    ax2.bar(x - width/2, df1['target_f1'], width, label=exp1_name, color=colors['exp1'], alpha=0.7)
    ax2.bar(x + width/2, df2['target_f1'], width, label=exp2_name, color=colors['exp2'], alpha=0.7)
    ax2.set_xlabel('Fold (Participant)')
    ax2.set_ylabel('Customized F1 Score')
    ax2.set_title('Final Performance Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'P{f}' for f in df1['fold']], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Inverse relative improvement comparison
    ax3 = axes[0, 2]
    ax3.bar(x - width/2, df1['inverse_relative_improvement'], width, label=exp1_name, color=colors['exp1'], alpha=0.7)
    ax3.bar(x + width/2, df2['inverse_relative_improvement'], width, label=exp2_name, color=colors['exp2'], alpha=0.7)
    ax3.set_xlabel('Fold (Participant)')
    ax3.set_ylabel('Inverse Relative Improvement (%)')
    ax3.set_title('Improvement Efficiency')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'P{f}' for f in df1['fold']], rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)

    # Plot 4: Paired differences
    ax4 = axes[1, 0]
    metrics_to_plot = ['target_f1', 'inverse_relative_improvement']
    metric_labels = ['Target F1', 'Inverse Relative Improvement (%)']

    for i, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        diffs = df2[metric].values - df1[metric].values
        ax4.scatter([i] * len(diffs), diffs, alpha=0.7, s=50,
                   color=colors['improve'] if np.mean(diffs) > 0 else colors['exp2'])
        ax4.scatter(i, np.mean(diffs), color='black', s=100, marker='_', linewidth=3)

    ax4.set_xlabel('Metric')
    ax4.set_ylabel(f'{exp2_name} - {exp1_name}')
    ax4.set_title('Paired Differences')
    ax4.set_xticks(range(len(metric_labels)))
    ax4.set_xticklabels(metric_labels, rotation=45)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.grid(True, alpha=0.3)

    # Plot 5: Effect sizes
    ax5 = axes[1, 1]
    effect_sizes = []
    metric_names = []
    colors_bar = []

    for result in results:
        if result['metric_key'] in ['base_f1', 'target_f1', 'inverse_relative_improvement']:
            effect_sizes.append(result['cohens_d'])
            metric_names.append(result['metric'].replace(' ', '\n'))

            # Color based on effect size magnitude and direction
            abs_d = abs(result['cohens_d'])
            if result['paired_p_value'] < 0.05:
                if abs_d >= 0.8:
                    colors_bar.append(colors['improve'] if result['cohens_d'] > 0 else colors['exp2'])
                else:
                    colors_bar.append(colors['neutral'])
            else:
                colors_bar.append('#cccccc')  # Light gray for non-significant

    bars = ax5.bar(range(len(effect_sizes)), effect_sizes, color=colors_bar, alpha=0.7)
    ax5.set_xlabel('Metric')
    ax5.set_ylabel("Cohen's d (Effect Size)")
    ax5.set_title('Effect Sizes')
    ax5.set_xticks(range(len(metric_names)))
    ax5.set_xticklabels(metric_names, rotation=45)
    ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax5.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5)
    ax5.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax5.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)
    ax5.axhline(y=-0.2, color='gray', linestyle='--', alpha=0.5)
    ax5.axhline(y=-0.5, color='gray', linestyle='--', alpha=0.5)
    ax5.axhline(y=-0.8, color='gray', linestyle='--', alpha=0.5)
    ax5.grid(True, alpha=0.3)

    # Plot 6: P-values summary
    ax6 = axes[1, 2]

    # Create a simple p-value significance plot
    metrics_for_pval = ['target_f1', 'inverse_relative_improvement']
    metric_labels_pval = ['Target F1', 'Inverse Rel. Imp.']
    p_values = []
    colors_pval = []

    for metric in metrics_for_pval:
        result = next((r for r in results if r['metric_key'] == metric), None)
        if result:
            p_val = result['paired_p_value']
            p_values.append(-np.log10(p_val))  # Convert to -log10 scale

            # Color based on significance
            if p_val < 0.001:
                colors_pval.append('#d62728')  # Red for highly significant
            elif p_val < 0.01:
                colors_pval.append('#ff7f0e')  # Orange for significant
            elif p_val < 0.05:
                colors_pval.append('#2ca02c')  # Green for marginally significant
            else:
                colors_pval.append('#7f7f7f')  # Gray for non-significant

    bars = ax6.bar(range(len(p_values)), p_values, color=colors_pval, alpha=0.7)
    ax6.set_xlabel('Metric')
    ax6.set_ylabel('-log₁₀(P-value)')
    ax6.set_title('Statistical Significance')
    ax6.set_xticks(range(len(metric_labels_pval)))
    ax6.set_xticklabels(metric_labels_pval, rotation=45)

    # Add significance threshold lines
    ax6.axhline(y=-np.log10(0.05), color='gray', linestyle='--', alpha=0.5, label='P=0.05')
    ax6.axhline(y=-np.log10(0.01), color='gray', linestyle='--', alpha=0.7, label='P=0.01')
    ax6.axhline(y=-np.log10(0.001), color='gray', linestyle='--', alpha=0.9, label='P=0.001')

    ax6.grid(True, alpha=0.3)
    ax6.legend(fontsize=8)

    # Row 3: Box and whisker plots for each key metric
    key_metrics_box = [
        ('base_f1', 'Base F1 Score'),
        ('target_f1', 'Customized F1 Score'),
        ('inverse_relative_improvement', 'Inverse Relative Improvement (%)')
    ]

    for i, (metric_key, metric_label) in enumerate(key_metrics_box):
        ax_box = axes[2, i]

        # Prepare data for box plots
        data_for_box = [df1[metric_key].values, df2[metric_key].values]
        labels_for_box = [exp1_name, exp2_name]

        # Create box plot
        bp = ax_box.boxplot(data_for_box, tick_labels=labels_for_box, patch_artist=True,
                           widths=0.6, medianprops=dict(color='white', linewidth=2))

        # Color the boxes
        bp['boxes'][0].set_facecolor(colors['exp1'])
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_facecolor(colors['exp2'])
        bp['boxes'][1].set_alpha(0.7)

        # Style whiskers and caps
        for whisker in bp['whiskers']:
            whisker.set_color(colors['neutral'])
            whisker.set_linewidth(1)
        for cap in bp['caps']:
            cap.set_color(colors['neutral'])
            cap.set_linewidth(1)

        # Add individual data points
        for j, data in enumerate(data_for_box):
            x_jitter = np.random.normal(j + 1, 0.04, len(data))
            ax_box.scatter(x_jitter, data, alpha=0.6, s=20,
                          color='white', edgecolors=colors['exp1'] if j == 0 else colors['exp2'],
                          linewidths=1)

        # Get statistical results for this metric
        stats_result = next((r for r in results if r['metric_key'] == metric_key), None)
        if stats_result:
            # Add p-value annotation
            p_val = stats_result['paired_p_value']
            if p_val < 0.001:
                p_text = "***"
            elif p_val < 0.01:
                p_text = "**"
            elif p_val < 0.05:
                p_text = "*"
            else:
                p_text = "ns"

            # Add significance annotation
            y_max = max(max(data_for_box[0]), max(data_for_box[1]))
            y_min = min(min(data_for_box[0]), min(data_for_box[1]))
            y_range = y_max - y_min

            # Draw significance bar
            y_sig = y_max + 0.05 * y_range
            ax_box.plot([1, 2], [y_sig, y_sig], 'k-', linewidth=1)
            ax_box.plot([1, 1], [y_sig, y_sig - 0.02 * y_range], 'k-', linewidth=1)
            ax_box.plot([2, 2], [y_sig, y_sig - 0.02 * y_range], 'k-', linewidth=1)
            ax_box.text(1.5, y_sig + 0.01 * y_range, p_text, ha='center', va='bottom', fontweight='bold')

        ax_box.set_ylabel(metric_label)
        ax_box.set_title(f'{metric_label} Distribution')
        ax_box.grid(True, alpha=0.3)
        ax_box.set_xticklabels(labels_for_box, rotation=45)

        # Special handling for inverse relative improvement (can be negative)
        if metric_key == 'inverse_relative_improvement':
            ax_box.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)

    plt.tight_layout()

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/experiment_comparison_{exp1_name}_vs_{exp2_name}.jpg"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n📈 Comparison plot saved as: {filename}")

    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Compare ML experiments statistically')
    parser.add_argument('exp1', help='First experiment name (e.g., medium)')
    parser.add_argument('exp2', help='Second experiment name (e.g., medium_augmented)')
    parser.add_argument('--output-dir', default='figures', help='Output directory for plots')
    parser.add_argument('--training-analysis', action='store_true',
                       help='Include deep training dynamics analysis for individual runs')
    parser.add_argument('--analyze-best', action='store_true',
                       help='Analyze training dynamics of best performing runs from each experiment')

    args = parser.parse_args()

    compare_experiments(args.exp1, args.exp2, args.output_dir, args.training_analysis, args.analyze_best)

if __name__ == "__main__":
    main()