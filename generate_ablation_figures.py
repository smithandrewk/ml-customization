#!/usr/bin/env python3
"""
Ablation Study Figure Generation Script

This script generates publication-quality figures from ablation study results.

Usage:
    python generate_ablation_figures.py /path/to/ablation_study_results
    python generate_ablation_figures.py /path/to/ablation_study_results --style publication
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plot parameters
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'serif'],
    'text.usetex': False,  # Set to True if you have LaTeX installed
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# Color schemes
COLORS = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#A23B72',    # Purple
    'accent': '#F18F01',       # Orange
    'success': '#C73E1D',      # Red
    'neutral': '#8B9A9C',      # Gray
    'baseline': '#666666',     # Dark gray for baseline
}

# Technique color mapping for consistency
TECHNIQUE_COLORS = {
    'baseline': COLORS['baseline'],
    'ewc_only': '#FF6B6B',
    'layerwise_only': '#4ECDC4', 
    'gradual_only': '#45B7D1',
    'augmentation_only': '#96CEB4',
    'coral_only': '#FFEAA7',
    'contrastive_only': '#DDA0DD',
    'ensemble_only': '#98D8C8',
    'ewc_layerwise': '#FF8C94',
    'ewc_augmentation': '#FFB3BA',
    'layerwise_augmentation': '#BFEFFF',
    'core_trio': '#FF6347',
    'domain_adaptation': '#20B2AA',
    'all_advanced': '#8A2BE2',
    'all_plus_ensemble': '#4B0082'
}

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate ablation study figures')
    parser.add_argument('study_dir', type=str, help='Path to ablation study results directory')
    parser.add_argument('--style', type=str, default='publication', choices=['publication', 'presentation'],
                       help='Figure style (publication=paper, presentation=slides)')
    parser.add_argument('--format', type=str, default='pdf', choices=['pdf', 'png', 'svg', 'eps'],
                       help='Output format')
    parser.add_argument('--dpi', type=int, default=300, help='Figure DPI')
    parser.add_argument('--figsize_scale', type=float, default=1.0, help='Scale factor for figure sizes')
    
    return parser.parse_args()

def load_analysis_data(study_dir):
    """Load the processed analysis data."""
    analysis_dir = Path(study_dir) / "analysis"
    
    # Load main results
    csv_file = analysis_dir / "ablation_results.csv"
    if not csv_file.exists():
        print(f"❌ Results file not found: {csv_file}")
        print("Please run analyze_ablation_results.py first")
        sys.exit(1)
    
    df = pd.read_csv(csv_file)
    
    # Load comprehensive analysis
    json_file = analysis_dir / "comprehensive_analysis.json"
    analysis_data = None
    if json_file.exists():
        with open(json_file, 'r') as f:
            analysis_data = json.load(f)
    
    return df, analysis_data

def create_technique_comparison_barplot(df, analysis_data, figsize=(12, 8)):
    """Create bar plot comparing technique performance."""
    # Group by experiment and compute statistics
    grouped = df.groupby('experiment')['absolute_improvement'].agg(['mean', 'std', 'sem']).reset_index()
    grouped = grouped.sort_values('mean', ascending=True)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bars with error bars
    bars = ax.barh(grouped['experiment'], grouped['mean'], 
                   xerr=grouped['sem'], capsize=4, alpha=0.8,
                   color=[TECHNIQUE_COLORS.get(exp, COLORS['primary']) for exp in grouped['experiment']])
    
    # Customize plot
    ax.set_xlabel('Mean Absolute F1 Improvement', fontweight='bold')
    ax.set_ylabel('Technique', fontweight='bold')
    ax.set_title('Ablation Study: Technique Performance Comparison', fontweight='bold', pad=20)
    
    # Add value labels on bars
    for i, (bar, mean_val, sem_val) in enumerate(zip(bars, grouped['mean'], grouped['sem'])):
        width = bar.get_width()
        label = f'{mean_val:.4f}±{sem_val:.4f}'
        ax.text(width + sem_val + 0.001, bar.get_y() + bar.get_height()/2, 
                label, ha='left', va='center', fontsize=9)
    
    # Add reference line at zero
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    # Grid for readability
    ax.grid(axis='x', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    return fig

def create_participant_variation_plot(df, figsize=(14, 8)):
    """Create plot showing variation across participants."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Left plot: Box plot by participant
    box_data = [df[df['participant'] == p]['absolute_improvement'].values 
                for p in sorted(df['participant'].unique())]
    
    bp = ax1.boxplot(box_data, labels=sorted(df['participant'].unique()), 
                     patch_artist=True, showmeans=True)
    
    # Color boxes
    for patch in bp['boxes']:
        patch.set_facecolor(COLORS['primary'])
        patch.set_alpha(0.7)
    
    ax1.set_xlabel('Participant', fontweight='bold')
    ax1.set_ylabel('Absolute F1 Improvement', fontweight='bold')
    ax1.set_title('Performance Variation by Participant', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Right plot: Heatmap of participant vs technique
    pivot_df = df.pivot_table(values='absolute_improvement', 
                             index='participant', 
                             columns='experiment', 
                             aggfunc='mean')
    
    # Reorder columns by mean performance
    col_order = pivot_df.mean().sort_values(ascending=False).index
    pivot_df = pivot_df[col_order]
    
    im = ax2.imshow(pivot_df.values, cmap='RdYlBu_r', aspect='auto')
    
    # Set ticks and labels
    ax2.set_xticks(range(len(pivot_df.columns)))
    ax2.set_xticklabels(pivot_df.columns, rotation=45, ha='right')
    ax2.set_yticks(range(len(pivot_df.index)))
    ax2.set_yticklabels(pivot_df.index)
    ax2.set_title('F1 Improvement Heatmap', fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label('Absolute F1 Improvement', fontweight='bold')
    
    # Add text annotations
    for i in range(len(pivot_df.index)):
        for j in range(len(pivot_df.columns)):
            value = pivot_df.iloc[i, j]
            if not np.isnan(value):
                text_color = 'white' if abs(value) > 0.02 else 'black'
                ax2.text(j, i, f'{value:.3f}', ha='center', va='center', 
                        color=text_color, fontsize=8)
    
    plt.tight_layout()
    return fig

def create_statistical_significance_plot(df, analysis_data, figsize=(12, 6)):
    """Create plot showing statistical significance of improvements."""
    if not analysis_data or 'statistical_tests' not in analysis_data:
        print("⚠️  No statistical test data available")
        return None
    
    stat_tests = analysis_data['statistical_tests']
    
    # Prepare data for plotting
    techniques = []
    mean_diffs = []
    p_values = []
    significant = []
    
    for technique, results in stat_tests.items():
        if 'mean_difference' in results and 'p_value' in results:
            techniques.append(technique)
            mean_diffs.append(results['mean_difference'])
            p_values.append(results['p_value'])
            significant.append(results.get('significant', False))
    
    if not techniques:
        print("⚠️  No valid statistical test results")
        return None
    
    # Sort by mean difference
    sorted_data = sorted(zip(techniques, mean_diffs, p_values, significant), 
                        key=lambda x: x[1], reverse=True)
    techniques, mean_diffs, p_values, significant = zip(*sorted_data)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Left plot: Mean difference with significance
    colors = [COLORS['success'] if sig else COLORS['neutral'] for sig in significant]
    bars = ax1.barh(range(len(techniques)), mean_diffs, color=colors, alpha=0.8)
    
    ax1.set_yticks(range(len(techniques)))
    ax1.set_yticklabels(techniques)
    ax1.set_xlabel('Mean Difference vs Baseline', fontweight='bold')
    ax1.set_title('Statistical Significance vs Baseline', fontweight='bold')
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax1.grid(axis='x', alpha=0.3)
    
    # Add significance indicators
    for i, (bar, p_val, is_sig) in enumerate(zip(bars, p_values, significant)):
        width = bar.get_width()
        label = f'p={p_val:.3f}{"*" if is_sig else ""}'
        x_pos = width + 0.001 if width >= 0 else width - 0.001
        ha = 'left' if width >= 0 else 'right'
        ax1.text(x_pos, i, label, ha=ha, va='center', fontsize=9)
    
    # Right plot: P-value visualization
    log_p_values = [-np.log10(p) for p in p_values]
    colors = [COLORS['success'] if p < 0.05 else COLORS['neutral'] for p in p_values]
    
    ax2.barh(range(len(techniques)), log_p_values, color=colors, alpha=0.8)
    ax2.set_yticks(range(len(techniques)))
    ax2.set_yticklabels(techniques)
    ax2.set_xlabel('-log₁₀(p-value)', fontweight='bold')
    ax2.set_title('Statistical Significance Levels', fontweight='bold')
    ax2.axvline(x=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='α=0.05')
    ax2.legend()
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_technique_category_analysis(df, figsize=(10, 8)):
    """Analyze techniques by category (individual vs combinations)."""
    # Categorize techniques
    individual_techniques = ['ewc_only', 'layerwise_only', 'gradual_only', 'augmentation_only', 
                           'coral_only', 'contrastive_only', 'ensemble_only']
    
    combination_techniques = ['ewc_layerwise', 'ewc_augmentation', 'layerwise_augmentation', 
                            'core_trio', 'domain_adaptation', 'all_advanced', 'all_plus_ensemble']
    
    df_copy = df.copy()
    df_copy['category'] = df_copy['experiment'].apply(
        lambda x: 'Individual' if x in individual_techniques 
        else 'Combination' if x in combination_techniques 
        else 'Baseline'
    )
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    # Top left: Category comparison
    category_stats = df_copy.groupby('category')['absolute_improvement'].agg(['mean', 'std', 'sem'])
    category_stats = category_stats.loc[['Baseline', 'Individual', 'Combination']]
    
    bars = ax1.bar(category_stats.index, category_stats['mean'], 
                   yerr=category_stats['sem'], capsize=5, alpha=0.8,
                   color=[COLORS['baseline'], COLORS['primary'], COLORS['secondary']])
    
    ax1.set_ylabel('Mean Absolute F1 Improvement', fontweight='bold')
    ax1.set_title('Performance by Technique Category', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, mean_val in zip(bars, category_stats['mean']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 0.001, 
                f'{mean_val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Top right: Distribution by category
    box_data = [df_copy[df_copy['category'] == cat]['absolute_improvement'].values 
                for cat in ['Baseline', 'Individual', 'Combination']]
    
    bp = ax2.boxplot(box_data, labels=['Baseline', 'Individual', 'Combination'], 
                     patch_artist=True, showmeans=True)
    
    colors = [COLORS['baseline'], COLORS['primary'], COLORS['secondary']]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('Absolute F1 Improvement', fontweight='bold')
    ax2.set_title('Distribution by Category', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Bottom left: Individual techniques detailed
    individual_df = df_copy[df_copy['category'] == 'Individual']
    if not individual_df.empty:
        individual_grouped = individual_df.groupby('experiment')['absolute_improvement'].mean().sort_values(ascending=False)
        
        bars = ax3.bar(range(len(individual_grouped)), individual_grouped.values, 
                       color=COLORS['primary'], alpha=0.8)
        ax3.set_xticks(range(len(individual_grouped)))
        ax3.set_xticklabels(individual_grouped.index, rotation=45, ha='right')
        ax3.set_ylabel('Mean Absolute F1 Improvement', fontweight='bold')
        ax3.set_title('Individual Techniques Ranking', fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
    
    # Bottom right: Combination techniques detailed
    combination_df = df_copy[df_copy['category'] == 'Combination']
    if not combination_df.empty:
        combination_grouped = combination_df.groupby('experiment')['absolute_improvement'].mean().sort_values(ascending=False)
        
        bars = ax4.bar(range(len(combination_grouped)), combination_grouped.values, 
                       color=COLORS['secondary'], alpha=0.8)
        ax4.set_xticks(range(len(combination_grouped)))
        ax4.set_xticklabels(combination_grouped.index, rotation=45, ha='right')
        ax4.set_ylabel('Mean Absolute F1 Improvement', fontweight='bold')
        ax4.set_title('Combination Techniques Ranking', fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_technique_impact_matrix(df, figsize=(12, 8)):
    """Create a matrix showing the impact of each individual technique."""
    # Define technique mapping
    technique_flags = {
        'EWC': 'use_ewc',
        'Layer-wise': 'use_layerwise_finetuning', 
        'Gradual': 'use_gradual_unfreezing',
        'Augmentation': 'use_augmentation',
        'CORAL': 'use_coral',
        'Contrastive': 'use_contrastive',
        'Ensemble': 'use_ensemble'
    }
    
    # Check which columns exist
    available_techniques = {name: col for name, col in technique_flags.items() 
                          if col in df.columns}
    
    if not available_techniques:
        print("⚠️  No technique flag columns found")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate impact for each technique
    impacts = {}
    
    for tech_name, tech_col in available_techniques.items():
        with_tech = df[df[tech_col] == True]['absolute_improvement']
        without_tech = df[df[tech_col] == False]['absolute_improvement']
        
        if len(with_tech) > 0 and len(without_tech) > 0:
            impact = with_tech.mean() - without_tech.mean()
            impacts[tech_name] = {
                'impact': impact,
                'with_mean': with_tech.mean(),
                'without_mean': without_tech.mean(),
                'with_count': len(with_tech),
                'without_count': len(without_tech)
            }
    
    if not impacts:
        print("⚠️  No technique impacts could be calculated")
        return None
    
    # Create visualization
    techniques = list(impacts.keys())
    impact_values = [impacts[tech]['impact'] for tech in techniques]
    
    # Sort by impact
    sorted_data = sorted(zip(techniques, impact_values), key=lambda x: x[1], reverse=True)
    techniques, impact_values = zip(*sorted_data)
    
    colors = [COLORS['success'] if val > 0 else COLORS['neutral'] for val in impact_values]
    bars = ax.barh(range(len(techniques)), impact_values, color=colors, alpha=0.8)
    
    ax.set_yticks(range(len(techniques)))
    ax.set_yticklabels(techniques)
    ax.set_xlabel('Mean Impact on F1 Improvement', fontweight='bold')
    ax.set_title('Individual Technique Impact Analysis', fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, tech, impact) in enumerate(zip(bars, techniques, impact_values)):
        width = bar.get_width()
        label = f'{impact:.4f}'
        x_pos = width + 0.0005 if width >= 0 else width - 0.0005
        ha = 'left' if width >= 0 else 'right'
        ax.text(x_pos, i, label, ha=ha, va='center', fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_summary_dashboard(df, analysis_data, figsize=(16, 12)):
    """Create a comprehensive summary dashboard."""
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1])
    
    # Top row: Overall performance metrics
    ax1 = fig.add_subplot(gs[0, :])
    
    # Performance comparison (top techniques)
    grouped = df.groupby('experiment')['absolute_improvement'].agg(['mean', 'sem']).reset_index()
    grouped = grouped.sort_values('mean', ascending=False).head(8)
    
    bars = ax1.bar(grouped['experiment'], grouped['mean'], yerr=grouped['sem'], 
                   capsize=4, alpha=0.8, color=COLORS['primary'])
    ax1.set_ylabel('Mean Absolute F1 Improvement', fontweight='bold')
    ax1.set_title('Top 8 Technique Performance', fontweight='bold', fontsize=16)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, mean_val in zip(bars, grouped['mean']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 0.0005, 
                f'{mean_val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Middle left: Category analysis
    ax2 = fig.add_subplot(gs[1, 0])
    df_copy = df.copy()
    individual_techniques = ['ewc_only', 'layerwise_only', 'gradual_only', 'augmentation_only', 
                           'coral_only', 'contrastive_only', 'ensemble_only']
    combination_techniques = ['ewc_layerwise', 'ewc_augmentation', 'layerwise_augmentation', 
                            'core_trio', 'domain_adaptation', 'all_advanced', 'all_plus_ensemble']
    
    df_copy['category'] = df_copy['experiment'].apply(
        lambda x: 'Individual' if x in individual_techniques 
        else 'Combination' if x in combination_techniques 
        else 'Baseline'
    )
    
    category_means = df_copy.groupby('category')['absolute_improvement'].mean()
    category_means = category_means.reindex(['Baseline', 'Individual', 'Combination'])
    
    wedges, texts, autotexts = ax2.pie(category_means.values, labels=category_means.index, 
                                      autopct='%1.1f%%', startangle=90,
                                      colors=[COLORS['baseline'], COLORS['primary'], COLORS['secondary']])
    ax2.set_title('Performance by Category', fontweight='bold')
    
    # Middle center: Participant variation
    ax3 = fig.add_subplot(gs[1, 1])
    participant_means = df.groupby('participant')['absolute_improvement'].mean().sort_values(ascending=False)
    
    ax3.bar(range(len(participant_means)), participant_means.values, 
            color=COLORS['accent'], alpha=0.8)
    ax3.set_xticks(range(len(participant_means)))
    ax3.set_xticklabels(participant_means.index, rotation=45)
    ax3.set_ylabel('Mean F1 Improvement', fontweight='bold')
    ax3.set_title('Performance by Participant', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # Middle right: Statistical significance count
    ax4 = fig.add_subplot(gs[1, 2])
    if analysis_data and 'statistical_tests' in analysis_data:
        stat_tests = analysis_data['statistical_tests']
        significant_count = sum(1 for results in stat_tests.values() 
                              if results.get('significant', False))
        total_tests = len(stat_tests)
        
        labels = ['Significant', 'Not Significant']
        sizes = [significant_count, total_tests - significant_count]
        colors = [COLORS['success'], COLORS['neutral']]
        
        wedges, texts, autotexts = ax4.pie(sizes, labels=labels, autopct='%1.0f', 
                                          colors=colors, startangle=90)
        ax4.set_title('Statistical Significance\n(vs Baseline)', fontweight='bold')
    
    # Bottom row: Key statistics
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    # Summary statistics
    overall_mean = df['absolute_improvement'].mean()
    overall_std = df['absolute_improvement'].std()
    best_technique = grouped.iloc[0]['experiment']
    best_improvement = grouped.iloc[0]['mean']
    
    stats_text = f"""
    ABLATION STUDY SUMMARY
    
    • Total Experiments: {len(df)}
    • Participants: {df['participant'].nunique()}
    • Techniques Tested: {df['experiment'].nunique()}
    
    • Overall Mean Improvement: {overall_mean:.4f} ± {overall_std:.4f} F1 score
    • Best Technique: {best_technique}
    • Best Improvement: {best_improvement:.4f} F1 score
    
    • Baseline F1: {df[df['experiment'] == 'baseline']['custom_model_test_f1'].mean():.4f} (mean)
    • Range of Improvements: [{df['absolute_improvement'].min():.4f}, {df['absolute_improvement'].max():.4f}]
    """
    
    ax5.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS['neutral'], alpha=0.1))
    
    plt.suptitle('Smoking Detection Customization: Ablation Study Results', 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    return fig

def main():
    args = parse_arguments()
    
    study_dir = Path(args.study_dir)
    if not study_dir.exists():
        print(f"❌ Study directory not found: {study_dir}")
        sys.exit(1)
    
    print(f"🎨 Generating ablation study figures: {study_dir}")
    
    # Update plot parameters based on style
    if args.style == 'presentation':
        plt.rcParams.update({
            'font.size': 14,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 13,
            'figure.titlesize': 18
        })
    
    # Load data
    df, analysis_data = load_analysis_data(study_dir)
    
    # Create figures directory
    figures_dir = study_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    # Scale figure sizes
    scale = args.figsize_scale
    
    print("📊 Generating figures...")
    
    # 1. Technique comparison bar plot
    print("   1. Technique comparison...")
    fig1 = create_technique_comparison_barplot(df, analysis_data, figsize=(12*scale, 8*scale))
    if fig1:
        fig1.savefig(figures_dir / f"technique_comparison.{args.format}", dpi=args.dpi)
        plt.close(fig1)
    
    # 2. Participant variation analysis
    print("   2. Participant variation...")
    fig2 = create_participant_variation_plot(df, figsize=(14*scale, 8*scale))
    if fig2:
        fig2.savefig(figures_dir / f"participant_variation.{args.format}", dpi=args.dpi)
        plt.close(fig2)
    
    # 3. Statistical significance plot
    print("   3. Statistical significance...")
    fig3 = create_statistical_significance_plot(df, analysis_data, figsize=(12*scale, 6*scale))
    if fig3:
        fig3.savefig(figures_dir / f"statistical_significance.{args.format}", dpi=args.dpi)
        plt.close(fig3)
    
    # 4. Technique category analysis
    print("   4. Category analysis...")
    fig4 = create_technique_category_analysis(df, figsize=(10*scale, 8*scale))
    if fig4:
        fig4.savefig(figures_dir / f"category_analysis.{args.format}", dpi=args.dpi)
        plt.close(fig4)
    
    # 5. Individual technique impact matrix
    print("   5. Technique impact matrix...")
    fig5 = create_technique_impact_matrix(df, figsize=(12*scale, 8*scale))
    if fig5:
        fig5.savefig(figures_dir / f"technique_impact_matrix.{args.format}", dpi=args.dpi)
        plt.close(fig5)
    
    # 6. Summary dashboard
    print("   6. Summary dashboard...")
    fig6 = create_summary_dashboard(df, analysis_data, figsize=(16*scale, 12*scale))
    if fig6:
        fig6.savefig(figures_dir / f"summary_dashboard.{args.format}", dpi=args.dpi)
        plt.close(fig6)
    
    print(f"\n✅ Figures generated successfully!")
    print(f"📁 Saved to: {figures_dir}")
    print(f"📄 Format: {args.format.upper()} at {args.dpi} DPI")
    
    # List generated files
    generated_files = list(figures_dir.glob(f"*.{args.format}"))
    print(f"\n📋 Generated files:")
    for file in sorted(generated_files):
        print(f"   • {file.name}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())