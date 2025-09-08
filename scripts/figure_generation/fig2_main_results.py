#!/usr/bin/env python3
"""
Figure 2: Main Results - LOPO Performance Comparison
===================================================

Generates the primary results figure showing:
- Panel A: Bar chart comparing base vs customized F1 scores per participant
- Panel B: Statistical significance visualization with effect sizes
- Panel C: Distribution of improvements across participants

This is the key figure demonstrating personalization effectiveness.

Usage:
    python scripts/figure_generation/fig2_main_results.py --results_dir results/lopo_6participants
    
Output:
    - figures/figure2_main_results.pdf - Publication-ready vector figure
    - figures/figure2_main_results.png - High-res raster for presentations
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

# Publication-quality settings
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'Arial',
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'legend.frameon': False,
    'figure.dpi': 300
})

def load_analysis_results(results_dir):
    """Load statistical analysis results and participant data."""
    # Load statistical analysis
    stats_file = os.path.join(results_dir, 'statistical_analysis.json')
    if not os.path.exists(stats_file):
        raise FileNotFoundError(f"Statistical analysis file not found: {stats_file}. Run analyze_results.py first.")
    
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    
    # Load participant analysis
    participant_file = os.path.join(results_dir, 'participant_analysis.csv')
    if not os.path.exists(participant_file):
        raise FileNotFoundError(f"Participant analysis file not found: {participant_file}. Run analyze_results.py first.")
    
    participant_df = pd.read_csv(participant_file)
    
    return stats, participant_df

def create_panel_a_performance_bars(ax, participant_df, stats):
    """Create Panel A: Bar chart comparing base vs customized F1 per participant."""
    participants = participant_df['participant'].values
    base_f1 = participant_df['base_f1'].values
    custom_f1 = participant_df['custom_f1'].values
    improvements = participant_df['f1_improvement'].values
    
    x = np.arange(len(participants))
    width = 0.35
    
    # Color coding: positive improvements in green, negative in red
    base_color = '#7f7f7f'  # Gray for base
    improvement_colors = ['#2ca02c' if imp > 0 else '#d62728' for imp in improvements]
    
    # Create bars
    bars1 = ax.bar(x - width/2, base_f1, width, label='Base Model', color=base_color, alpha=0.8)
    bars2 = ax.bar(x + width/2, custom_f1, width, label='Customized Model', color=improvement_colors, alpha=0.8)
    
    # Add value labels on bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        
        # Base F1 label
        ax.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.005,
                f'{height1:.3f}', ha='center', va='bottom', fontsize=9, color='black')
        
        # Custom F1 label with improvement
        improvement_text = f'{height2:.3f}\n({improvements[i]:+.3f})'
        ax.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.005,
                improvement_text, ha='center', va='bottom', fontsize=9, 
                color='darkgreen' if improvements[i] > 0 else 'darkred', weight='bold')
    
    # Formatting
    ax.set_xlabel('Participants', fontweight='bold')
    ax.set_ylabel('F1 Score', fontweight='bold')
    ax.set_title('A. Individual Participant Performance', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(participants, rotation=45, ha='right')
    ax.set_ylim(0, max(max(base_f1), max(custom_f1)) * 1.15)
    
    # Legend
    ax.legend(loc='upper left', fontsize=10)
    
    # Add horizontal grid
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

def create_panel_b_statistics(ax, stats):
    """Create Panel B: Statistical significance and effect size visualization."""
    # Extract key statistics
    mean_improvement = stats['descriptive_stats']['improvement']['mean']
    ci_lower, ci_upper = stats['statistical_tests']['confidence_intervals']['mean_improvement_95ci']
    p_value = stats['statistical_tests']['paired_ttest']['p_value']
    cohens_d = stats['statistical_tests']['effect_size']['cohens_d']
    n_positive = stats['statistical_tests']['improvement_patterns']['positive_count']
    n_total = stats['n_participants']
    
    # Create effect size visualization
    y_pos = [3, 2, 1, 0]
    labels = [
        f'Mean Improvement\n{mean_improvement:.4f} F1 points',
        f'95% Confidence Interval\n[{ci_lower:.4f}, {ci_upper:.4f}]',
        f'Statistical Significance\np = {p_value:.4f}',
        f'Effect Size (Cohen\'s d)\n{cohens_d:.4f} ({stats["statistical_tests"]["effect_size"]["interpretation"]})'
    ]
    
    # Color coding based on significance and effect size
    colors = [
        '#2ca02c' if mean_improvement > 0 else '#d62728',  # Green if positive
        '#1f77b4',  # Blue for CI
        '#2ca02c' if p_value < 0.05 else '#ff7f0e',  # Green if significant
        '#2ca02c' if abs(cohens_d) > 0.5 else '#ff7f0e'  # Green if large effect
    ]
    
    # Create horizontal bar chart
    values = [mean_improvement, ci_upper - ci_lower, -np.log10(p_value), cohens_d]
    bars = ax.barh(y_pos, [abs(v) for v in values], color=colors, alpha=0.7)
    
    # Add labels
    for i, (bar, label, value) in enumerate(zip(bars, labels, values)):
        width = bar.get_width()
        ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                label, ha='left', va='center', fontsize=10, weight='bold')
    
    # Formatting
    ax.set_xlabel('Magnitude', fontweight='bold')
    ax.set_title('B. Statistical Summary', fontweight='bold', pad=15)
    ax.set_yticks([])
    ax.set_xlim(0, max([abs(v) for v in values]) * 2)
    
    # Add significance indicators
    significance_text = "✓ Significant" if p_value < 0.05 else "✗ Not Significant"
    effect_text = f"✓ {stats['statistical_tests']['effect_size']['interpretation'].title()} Effect" if abs(cohens_d) > 0.2 else "✗ Negligible Effect"
    
    ax.text(0.98, 0.95, f"{significance_text}\n{effect_text}\n{n_positive}/{n_total} participants improved", 
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

def create_panel_c_improvement_distribution(ax, participant_df, stats):
    """Create Panel C: Distribution of F1 improvements."""
    improvements = participant_df['f1_improvement'].values
    participants = participant_df['participant'].values
    
    # Create scatter plot with participant labels
    colors = ['#2ca02c' if imp > 0 else '#d62728' if imp < 0 else '#7f7f7f' for imp in improvements]
    
    scatter = ax.scatter(range(len(participants)), improvements, 
                        c=colors, s=100, alpha=0.8, edgecolors='black', linewidth=1)
    
    # Add participant labels
    for i, (participant, improvement) in enumerate(zip(participants, improvements)):
        offset = 0.008 if improvement >= 0 else -0.015
        ax.annotate(participant, (i, improvement + offset), 
                   ha='center', va='bottom' if improvement >= 0 else 'top',
                   fontsize=9, weight='bold')
    
    # Add horizontal line at zero
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.8)
    
    # Add mean line with confidence interval
    mean_improvement = stats['descriptive_stats']['improvement']['mean']
    ci_lower, ci_upper = stats['statistical_tests']['confidence_intervals']['mean_improvement_95ci']
    
    ax.axhline(y=mean_improvement, color='blue', linestyle='--', linewidth=2, alpha=0.8, label=f'Mean: {mean_improvement:.4f}')
    ax.fill_between(range(len(participants)), ci_lower, ci_upper, alpha=0.2, color='blue', label='95% CI')
    
    # Formatting
    ax.set_xlabel('Participants', fontweight='bold')
    ax.set_ylabel('F1 Improvement', fontweight='bold')
    ax.set_title('C. Distribution of Improvements', fontweight='bold', pad=15)
    ax.set_xticks(range(len(participants)))
    ax.set_xticklabels(participants, rotation=45, ha='right')
    
    # Grid and legend
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', fontsize=10)
    
    # Color legend
    positive_patch = mpatches.Patch(color='#2ca02c', label='Positive')
    negative_patch = mpatches.Patch(color='#d62728', label='Negative')
    ax.legend(handles=[positive_patch, negative_patch], loc='upper left', fontsize=10, title='Improvement')

def create_figure_2(stats, participant_df, output_dir):
    """Create the complete Figure 2 with all panels."""
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax_empty)) = plt.subplots(2, 2, figsize=(16, 12))
    ax_empty.axis('off')  # Hide the empty subplot
    
    # Create panels
    create_panel_a_performance_bars(ax1, participant_df, stats)
    create_panel_b_statistics(ax2, stats)
    create_panel_c_improvement_distribution(ax3, participant_df, stats)
    
    # Add main title
    fig.suptitle('Personalized Health Monitoring: Leave-One-Participant-Out Results', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Add summary text box
    n_participants = len(participant_df)
    n_positive = stats['statistical_tests']['improvement_patterns']['positive_count']
    mean_improvement = stats['descriptive_stats']['improvement']['mean']
    p_value = stats['statistical_tests']['paired_ttest']['p_value']
    cohens_d = stats['statistical_tests']['effect_size']['cohens_d']
    
    summary_text = f"""Key Findings:
    • {n_positive}/{n_participants} participants improved with customization
    • Mean improvement: {mean_improvement:+.4f} F1 points
    • Statistical significance: p = {p_value:.4f}
    • Effect size: d = {cohens_d:.4f} ({stats['statistical_tests']['effect_size']['interpretation']})"""
    
    fig.text(0.75, 0.25, summary_text, fontsize=11, 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
             verticalalignment='top')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    
    # Save figures
    os.makedirs(output_dir, exist_ok=True)
    
    # Vector format for publication
    pdf_path = os.path.join(output_dir, 'figure2_main_results.pdf')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    
    # Raster format for presentations
    png_path = os.path.join(output_dir, 'figure2_main_results.png')
    fig.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    
    plt.close()
    
    return pdf_path, png_path

def main():
    parser = argparse.ArgumentParser(description='Generate Figure 2: Main LOPO Results')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing analysis results')
    parser.add_argument('--output_dir', type=str, default='figures',
                       help='Output directory for figures (default: figures/)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.results_dir):
        print(f"❌ Error: Results directory {args.results_dir} does not exist")
        sys.exit(1)
    
    print(f"📊 GENERATING FIGURE 2: MAIN RESULTS")
    print(f"📂 Results directory: {args.results_dir}")
    print(f"🎨 Output directory: {args.output_dir}")
    
    try:
        # Load analysis results
        print(f"\n📥 Loading analysis results...")
        stats, participant_df = load_analysis_results(args.results_dir)
        
        # Generate figure
        print(f"\n🎨 Creating Figure 2...")
        pdf_path, png_path = create_figure_2(stats, participant_df, args.output_dir)
        
        print(f"\n✅ FIGURE 2 GENERATED SUCCESSFULLY")
        print(f"📄 PDF (publication): {pdf_path}")
        print(f"🖼️  PNG (presentation): {png_path}")
        
        # Summary of what was plotted
        n_participants = len(participant_df)
        n_positive = stats['statistical_tests']['improvement_patterns']['positive_count']
        mean_improvement = stats['descriptive_stats']['improvement']['mean']
        
        print(f"\n📈 Figure Summary:")
        print(f"   • Participants: {n_participants}")
        print(f"   • Positive improvements: {n_positive}/{n_participants}")
        print(f"   • Mean improvement: {mean_improvement:+.4f} F1 points")
        print(f"   • Statistical significance: {'YES' if stats['statistical_tests']['paired_ttest']['significant'] else 'NO'}")
        
    except Exception as e:
        print(f"❌ Error generating figure: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()