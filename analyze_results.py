"""
Analyze results from multiple two-phase customization experiments.

This script collects metrics from all experiment directories and creates
summary statistics and visualizations for paper results.
"""

import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def collect_experiment_results(experiments_dir='experiments'):
    """
    Collect results from all customization experiments.
    
    Args:
        experiments_dir: Directory containing experiment folders
    
    Returns:
        DataFrame with results from all participants
    """
    results = []
    
    # Find all experiment directories with 'custom' in the name
    for exp_dir in os.listdir(experiments_dir):
        if 'custom_' in exp_dir:
            exp_path = os.path.join(experiments_dir, exp_dir)
            metrics_path = os.path.join(exp_path, 'custom_metrics.pt')
            
            if os.path.exists(metrics_path):
                try:
                    metrics = torch.load(metrics_path, map_location='cpu', weights_only=False)
                    
                    result = {
                        'experiment_dir': exp_dir,
                        'participant': metrics['target_participant'],
                        'base_test_f1': metrics['base_model_test_f1'],
                        'custom_test_f1': metrics['custom_model_test_f1'],
                        'absolute_improvement': metrics['absolute_improvement'],
                        'percentage_improvement': metrics['percentage_improvement'],
                        'base_best_epoch': metrics['base_best_epoch'],
                        'custom_best_epoch': metrics['custom_best_epoch'],
                        'base_val_metric': metrics.get('best_val_metric', metrics.get('best_target_val_f1', None)),  # Phase 1 validation metric
                        'custom_val_metric': metrics.get('best_target_val_metric', metrics.get('best_target_val_f1', None)),  # Phase 2 validation metric
                        'base_metric_name': metrics.get('base_metric_name', 'F1'),
                        'custom_metric_name': metrics.get('custom_metric_name', 'F1'),
                        'early_stopping_metric': metrics.get('early_stopping_metric', 'f1'),
                        'evaluation_method': metrics.get('evaluation_method', 'moving_average'),
                        'ma_window_size': metrics.get('ma_window_size', 10),
                        'custom_learning_rate': metrics.get('custom_learning_rate', None),
                        'combined_train_samples': metrics.get('combined_train_samples', None),
                        'target_val_samples': metrics.get('target_val_samples', None),
                        'target_test_samples': metrics.get('target_test_samples', None),
                        'base_total_epochs': metrics.get('base_total_epochs', None),
                        'custom_total_epochs': metrics.get('custom_total_epochs', None),
                        'target_weight_multiplier': metrics.get('target_weight_multiplier', 1.0),
                        'effective_target_percentage': metrics.get('effective_target_percentage', None)
                    }
                    
                    results.append(result)
                    print(f"✓ Loaded: {exp_dir} - {metrics['target_participant']}")
                    
                except Exception as e:
                    print(f"✗ Error loading {exp_dir}: {e}")
            else:
                print(f"✗ Missing metrics file: {metrics_path}")
    
    if not results:
        print("No experiment results found!")
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    print(f"\n📊 Collected results from {len(df)} experiments")
    return df

def generate_summary_statistics(df):
    """Generate summary statistics for the results."""
    if df.empty:
        return
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Overall statistics
    print(f"Total participants: {len(df)}")
    print(f"Evaluation method: {df['evaluation_method'].iloc[0] if len(df['evaluation_method'].unique()) == 1 else 'Mixed'}")
    print(f"Early stopping metric: {df['early_stopping_metric'].iloc[0] if len(df['early_stopping_metric'].unique()) == 1 else 'Mixed'}")
    if df['ma_window_size'].notna().any():
        print(f"Moving average window: {df['ma_window_size'].iloc[0]}")
    
    # Target weighting summary
    if df['target_weight_multiplier'].notna().any():
        unique_weights = df['target_weight_multiplier'].unique()
        if len(unique_weights) == 1:
            print(f"Target weight multiplier: {unique_weights[0]:.1f}")
            if df['effective_target_percentage'].notna().any():
                print(f"Effective target representation: {df['effective_target_percentage'].mean():.1f}% ± {df['effective_target_percentage'].std():.1f}%")
        else:
            print(f"Target weight multiplier: Mixed ({unique_weights})")
            if df['effective_target_percentage'].notna().any():
                print(f"Effective target representation: {df['effective_target_percentage'].mean():.1f}% ± {df['effective_target_percentage'].std():.1f}%")
    
    # Training epochs summary
    if df['base_total_epochs'].notna().any():
        print(f"\nTraining Epochs:")
        print(f"  Phase 1 (Base): {df['base_total_epochs'].mean():.1f} ± {df['base_total_epochs'].std():.1f} epochs")
        if df['custom_total_epochs'].notna().any():
            print(f"  Phase 2 (Custom): {df['custom_total_epochs'].mean():.1f} ± {df['custom_total_epochs'].std():.1f} epochs")
            print(f"  Total training: {(df['base_total_epochs'] + df['custom_total_epochs']).mean():.1f} ± {(df['base_total_epochs'] + df['custom_total_epochs']).std():.1f} epochs")
    
    print(f"\nBase Model Performance:")
    print(f"  Mean F1: {df['base_test_f1'].mean():.4f} ± {df['base_test_f1'].std():.4f}")
    print(f"  Min F1:  {df['base_test_f1'].min():.4f}")
    print(f"  Max F1:  {df['base_test_f1'].max():.4f}")
    
    print(f"\nCustomized Model Performance:")
    print(f"  Mean F1: {df['custom_test_f1'].mean():.4f} ± {df['custom_test_f1'].std():.4f}")
    print(f"  Min F1:  {df['custom_test_f1'].min():.4f}")
    print(f"  Max F1:  {df['custom_test_f1'].max():.4f}")
    
    print(f"\nImprovement Statistics:")
    print(f"  Mean absolute improvement: {df['absolute_improvement'].mean():+.4f} ± {df['absolute_improvement'].std():.4f}")
    print(f"  Mean percentage improvement: {df['percentage_improvement'].mean():+.2f}% ± {df['percentage_improvement'].std():.2f}%")
    print(f"  Participants improved: {(df['absolute_improvement'] > 0).sum()}/{len(df)} ({(df['absolute_improvement'] > 0).mean()*100:.1f}%)")
    print(f"  Max improvement: {df['absolute_improvement'].max():+.4f} ({df['percentage_improvement'].max():+.2f}%)")
    print(f"  Min improvement: {df['absolute_improvement'].min():+.4f} ({df['percentage_improvement'].min():+.2f}%)")
    
    # Statistical significance test
    from scipy import stats
    if len(df) > 1:
        t_stat, p_value = stats.ttest_rel(df['custom_test_f1'], df['base_test_f1'])
        print(f"\nPaired t-test (custom vs base):")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.6f}")
        print(f"  Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")

def create_visualizations(df, save_dir='results_analysis'):
    """Create visualizations of the results."""
    if df.empty:
        return
    
    os.makedirs(save_dir, exist_ok=True)
    plt.style.use('default')
    
    # Set up the plotting style
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11
    })
    
    # 1. Before vs After Performance
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Scatter plot: Before vs After
    ax1 = axes[0, 0]
    ax1.scatter(df['base_test_f1'], df['custom_test_f1'], alpha=0.7, s=60)
    
    # Add diagonal line (no improvement)
    min_f1 = min(df['base_test_f1'].min(), df['custom_test_f1'].min())
    max_f1 = max(df['base_test_f1'].max(), df['custom_test_f1'].max())
    ax1.plot([min_f1, max_f1], [min_f1, max_f1], 'r--', alpha=0.7, label='No improvement')
    
    ax1.set_xlabel('Base Model F1')
    ax1.set_ylabel('Customized Model F1')
    ax1.set_title('Base vs Customized Model Performance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add participant labels
    for i, row in df.iterrows():
        ax1.annotate(row['participant'], (row['base_test_f1'], row['custom_test_f1']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
    
    # 2. Improvement Distribution
    ax2 = axes[0, 1]
    ax2.hist(df['absolute_improvement'], bins=min(10, len(df)), alpha=0.7, edgecolor='black')
    ax2.axvline(0, color='red', linestyle='--', alpha=0.7, label='No improvement')
    ax2.set_xlabel('Absolute F1 Improvement')
    ax2.set_ylabel('Number of Participants')
    ax2.set_title('Distribution of F1 Improvements')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Percentage Improvement
    ax3 = axes[1, 0]
    bars = ax3.bar(range(len(df)), df['percentage_improvement'], 
                   color=['green' if x > 0 else 'red' for x in df['percentage_improvement']], alpha=0.7)
    ax3.axhline(0, color='black', linestyle='-', alpha=0.5)
    ax3.set_xlabel('Participant')
    ax3.set_ylabel('Percentage Improvement (%)')
    ax3.set_title('Percentage F1 Improvement by Participant')
    ax3.set_xticks(range(len(df)))
    ax3.set_xticklabels(df['participant'], rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, df['percentage_improvement'])):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -3),
                f'{val:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
    
    # 4. Box plot comparison
    ax4 = axes[1, 1]
    data_to_plot = [df['base_test_f1'], df['custom_test_f1']]
    box_plot = ax4.boxplot(data_to_plot, labels=['Base Model', 'Customized Model'], patch_artist=True)
    box_plot['boxes'][0].set_facecolor('lightblue')
    box_plot['boxes'][1].set_facecolor('lightgreen')
    ax4.set_ylabel('F1 Score')
    ax4.set_title('Performance Comparison')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Individual participant trajectories (if we have epoch data)
    if 'base_best_epoch' in df.columns and 'custom_best_epoch' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for i, row in df.iterrows():
            # Plot line from base to custom performance
            ax.plot([0, 1], [row['base_test_f1'], row['custom_test_f1']], 
                   'o-', alpha=0.7, linewidth=2, markersize=8, label=row['participant'])
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Base Model\n(Phase 1)', 'Customized Model\n(Phase 2)'])
        ax.set_ylabel('F1 Score')
        ax.set_title('Individual Participant Performance Trajectories')
        ax.grid(True, alpha=0.3)
        
        # Add legend (but limit to reasonable number of entries)
        if len(df) <= 15:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/participant_trajectories.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"\n📈 Visualizations saved to: {save_dir}/")

def create_best_models_table(df, save_dir='results_analysis'):
    """Create a summary table of best models from both phases."""
    if df.empty:
        return
    
    print("\n" + "="*80)
    print("BEST MODELS SUMMARY TABLE")
    print("="*80)
    
    # Create table data
    table_data = []
    for _, row in df.iterrows():
        table_data.append({
            'Participant': row['participant'],
            'Phase 1 Epoch': int(row['base_best_epoch']) if pd.notna(row['base_best_epoch']) else 'N/A',
            'Phase 1 Metric': f"{row['base_val_metric']:.4f}" if pd.notna(row['base_val_metric']) else 'N/A',
            'Phase 1 Test F1': f"{row['base_test_f1']:.4f}",
            'Phase 2 Epoch': int(row['custom_best_epoch']) if pd.notna(row['custom_best_epoch']) else 'N/A',
            'Phase 2 Metric': f"{row['custom_val_metric']:.4f}" if pd.notna(row['custom_val_metric']) else 'N/A', 
            'Phase 2 Test F1': f"{row['custom_test_f1']:.4f}",
            'Target Weight': f"{row['target_weight_multiplier']:.1f}" if pd.notna(row['target_weight_multiplier']) else '1.0',
            'Improvement': f"{row['absolute_improvement']:+.4f} ({row['percentage_improvement']:+.2f}%)"
        })
    
    # Print table
    headers = list(table_data[0].keys())
    col_widths = {h: max(len(h), max(len(str(row[h])) for row in table_data)) for h in headers}
    
    # Header
    header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
    print(header_line)
    print("-" * len(header_line))
    
    # Rows
    for row in table_data:
        row_line = " | ".join(str(row[h]).ljust(col_widths[h]) for h in headers)
        print(row_line)
    
    # Save to CSV
    os.makedirs(save_dir, exist_ok=True)
    best_models_df = pd.DataFrame(table_data)
    csv_path = f'{save_dir}/best_models_summary.csv'
    best_models_df.to_csv(csv_path, index=False)
    print(f"\n📄 Best models summary saved to: {csv_path}")

def save_results_csv(df, save_dir='results_analysis'):
    """Save results to CSV for further analysis."""
    if df.empty:
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Main results CSV
    csv_path = f'{save_dir}/experiment_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"📄 Results saved to: {csv_path}")
    
    # Summary statistics CSV
    summary_stats = {
        'Metric': [
            'Participants', 'Base F1 Mean', 'Base F1 Std', 'Base F1 Min', 'Base F1 Max',
            'Custom F1 Mean', 'Custom F1 Std', 'Custom F1 Min', 'Custom F1 Max',
            'Abs Improvement Mean', 'Abs Improvement Std', 'Abs Improvement Min', 'Abs Improvement Max',
            'Pct Improvement Mean', 'Pct Improvement Std', 'Pct Improvement Min', 'Pct Improvement Max',
            'Participants Improved', 'Improvement Rate (%)'
        ],
        'Value': [
            len(df),
            df['base_test_f1'].mean(), df['base_test_f1'].std(), df['base_test_f1'].min(), df['base_test_f1'].max(),
            df['custom_test_f1'].mean(), df['custom_test_f1'].std(), df['custom_test_f1'].min(), df['custom_test_f1'].max(),
            df['absolute_improvement'].mean(), df['absolute_improvement'].std(), 
            df['absolute_improvement'].min(), df['absolute_improvement'].max(),
            df['percentage_improvement'].mean(), df['percentage_improvement'].std(), 
            df['percentage_improvement'].min(), df['percentage_improvement'].max(),
            (df['absolute_improvement'] > 0).sum(), (df['absolute_improvement'] > 0).mean() * 100
        ]
    }
    
    summary_df = pd.DataFrame(summary_stats)
    summary_path = f'{save_dir}/summary_statistics.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"📊 Summary statistics saved to: {summary_path}")

def main():
    """Main analysis function."""
    print("🔍 Analyzing Two-Phase Customization Results")
    print("=" * 50)
    
    # Collect results
    df = collect_experiment_results()
    
    if df.empty:
        print("❌ No results found to analyze!")
        return
    
    # Generate summary statistics
    generate_summary_statistics(df)
    
    # Create best models summary table
    create_best_models_table(df)
    
    # Create visualizations
    create_visualizations(df)
    
    # Save results
    save_results_csv(df)
    
    print(f"\n✅ Analysis complete! Results from {len(df)} participants analyzed.")
    
    # Print top performers
    if len(df) > 0:
        print(f"\n🏆 Top Performers:")
        top_df = df.nlargest(3, 'absolute_improvement')[['participant', 'base_test_f1', 'custom_test_f1', 'absolute_improvement', 'percentage_improvement']]
        for _, row in top_df.iterrows():
            print(f"  {row['participant']}: {row['base_test_f1']:.4f} → {row['custom_test_f1']:.4f} ({row['absolute_improvement']:+.4f}, {row['percentage_improvement']:+.2f}%)")

if __name__ == "__main__":
    main()