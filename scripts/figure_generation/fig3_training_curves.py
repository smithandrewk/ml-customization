#!/usr/bin/env python3
"""
Figure 3: Training Curves - Two-Phase Training Visualization
===========================================================

Generates training curve visualizations showing:
- Panel A: Combined training curves for all participants (phase transitions)
- Panel B: Individual participant training curves (grid layout)
- Panel C: Phase transition analysis (base vs custom performance)

Shows the two-phase training methodology and individual variability.

Usage:
    python scripts/figure_generation/fig3_training_curves.py --results_dir results/lopo_6participants
    
Output:
    - figures/figure3_training_curves.pdf - Publication-ready vector figure
    - figures/figure3_training_curves.png - High-res raster for presentations
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path
import glob

# Publication-quality settings
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'Arial',
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'legend.frameon': False,
    'figure.dpi': 300
})

def load_participant_training_data(results_dir):
    """Load training metrics for all participants."""
    participant_data = {}
    
    # Look for participant directories
    for participant_dir in os.listdir(results_dir):
        participant_path = os.path.join(results_dir, participant_dir)
        if os.path.isdir(participant_path):
            # Look for metrics files
            base_metrics_file = os.path.join(participant_path, 'base_metrics.pt')
            custom_metrics_file = os.path.join(participant_path, 'custom_metrics.pt')
            
            if os.path.exists(base_metrics_file) and os.path.exists(custom_metrics_file):
                try:
                    base_metrics = torch.load(base_metrics_file, map_location='cpu')
                    custom_metrics = torch.load(custom_metrics_file, map_location='cpu')
                    
                    participant_data[participant_dir] = {
                        'base_metrics': base_metrics,
                        'custom_metrics': custom_metrics
                    }
                except Exception as e:
                    print(f"Warning: Could not load metrics for {participant_dir}: {e}")
    
    return participant_data

def create_panel_a_combined_curves(ax, participant_data):
    """Create Panel A: Combined training curves showing all participants."""
    colors = plt.cm.Set1(np.linspace(0, 1, len(participant_data)))
    
    for i, (participant, data) in enumerate(participant_data.items()):
        base_metrics = data['base_metrics']
        custom_metrics = data['custom_metrics']
        
        # Extract training data
        base_epochs = len(base_metrics['train_f1'])
        custom_epochs = len(custom_metrics['train_f1'])
        
        # Create continuous x-axis
        base_x = np.arange(base_epochs)
        custom_x = np.arange(base_epochs, base_epochs + custom_epochs)
        
        # Plot base phase (target performance during base training)
        # We'll use the test_f1 from base metrics as proxy for target performance
        if 'test_f1' in base_metrics:
            base_target_f1 = base_metrics['test_f1']
        else:
            # Fallback if not available
            base_target_f1 = [0.5] * base_epochs
        
        # Plot custom phase (target performance during customization)
        custom_target_f1 = custom_metrics.get('target_test_f1', custom_metrics.get('train_f1', []))
        
        # Plot combined curve
        combined_x = np.concatenate([base_x, custom_x])
        combined_f1 = np.concatenate([base_target_f1, custom_target_f1])
        
        ax.plot(combined_x, combined_f1, color=colors[i], linewidth=2, alpha=0.7, 
                label=f'{participant}')
        
        # Add phase transition line
        ax.axvline(x=base_epochs, color=colors[i], linestyle='--', alpha=0.5, linewidth=1)
    
    # Add single phase transition marker
    if participant_data:
        first_participant = list(participant_data.values())[0]
        transition_epoch = len(first_participant['base_metrics']['train_f1'])
        ax.axvline(x=transition_epoch, color='black', linestyle='--', linewidth=2, 
                  alpha=0.8, label='Phase Transition')
    
    # Formatting
    ax.set_xlabel('Training Epochs', fontweight='bold')
    ax.set_ylabel('Target F1 Score', fontweight='bold')
    ax.set_title('A. Combined Training Curves (All Participants)', fontweight='bold', pad=15)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1)
    
    # Add phase labels
    if participant_data:
        transition_epoch = len(list(participant_data.values())[0]['base_metrics']['train_f1'])
        ax.text(transition_epoch//2, 0.95, 'Phase 1:\nBase Training', 
               ha='center', va='top', fontsize=10, weight='bold',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        ax.text(transition_epoch + (ax.get_xlim()[1] - transition_epoch)//2, 0.95, 
               'Phase 2:\nCustomization', ha='center', va='top', fontsize=10, weight='bold',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

def create_panel_b_individual_curves(fig, participant_data):
    """Create Panel B: Individual training curves in grid layout."""
    n_participants = len(participant_data)
    if n_participants == 0:
        return
    
    # Calculate grid dimensions
    n_cols = min(3, n_participants)
    n_rows = (n_participants + n_cols - 1) // n_cols
    
    # Create subplots for individual curves
    gs = fig.add_gridspec(n_rows, n_cols, left=0.1, right=0.95, top=0.45, bottom=0.05, 
                         hspace=0.4, wspace=0.3)
    
    for i, (participant, data) in enumerate(participant_data.items()):
        row = i // n_cols
        col = i % n_cols
        ax = fig.add_subplot(gs[row, col])
        
        base_metrics = data['base_metrics']
        custom_metrics = data['custom_metrics']
        
        # Extract data
        base_epochs = len(base_metrics['train_f1'])
        custom_epochs = len(custom_metrics.get('train_f1', []))
        
        # Plot base phase
        base_x = np.arange(base_epochs)
        if 'test_f1' in base_metrics:
            ax.plot(base_x, base_metrics['test_f1'], color='#1f77b4', linewidth=2, 
                   label='Base (Test)', alpha=0.8)
        
        # Plot custom phase  
        if custom_epochs > 0:
            custom_x = np.arange(base_epochs, base_epochs + custom_epochs)
            target_f1 = custom_metrics.get('target_test_f1', custom_metrics.get('train_f1', []))
            if len(target_f1) > 0:
                ax.plot(custom_x, target_f1, color='#ff7f0e', linewidth=2, 
                       label='Custom (Target)', alpha=0.8)
        
        # Phase transition
        ax.axvline(x=base_epochs, color='gray', linestyle='--', alpha=0.6)
        
        # Formatting
        ax.set_title(f'{participant}', fontweight='bold', fontsize=11)
        ax.set_xlabel('Epochs', fontsize=9)
        ax.set_ylabel('F1 Score', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        if i == 0:  # Add legend to first subplot
            ax.legend(fontsize=8)
    
    # Add panel title
    fig.text(0.5, 0.5, 'B. Individual Participant Training Curves', 
             ha='center', va='top', fontsize=14, fontweight='bold')

def create_panel_c_phase_analysis(ax, participant_data):
    """Create Panel C: Phase transition analysis."""
    participants = []
    base_final_f1 = []
    custom_final_f1 = []
    improvements = []
    
    for participant, data in participant_data.items():
        base_metrics = data['base_metrics']
        custom_metrics = data['custom_metrics']
        
        # Get final F1 scores
        base_f1 = base_metrics.get('best_f1', base_metrics.get('test_f1', [0])[-1] if base_metrics.get('test_f1') else 0)
        custom_f1 = custom_metrics.get('best_target_f1', 0)
        
        participants.append(participant)
        base_final_f1.append(base_f1)
        custom_final_f1.append(custom_f1)
        improvements.append(custom_f1 - base_f1)
    
    if not participants:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Create improvement analysis scatter plot
    x = np.arange(len(participants))
    
    # Color code improvements
    colors = ['#2ca02c' if imp > 0 else '#d62728' if imp < 0 else '#7f7f7f' for imp in improvements]
    
    # Plot base performance as baseline
    ax.bar(x, base_final_f1, alpha=0.5, color='#1f77b4', label='Base Performance')
    
    # Plot improvements as overlaid bars
    positive_improvements = [max(0, imp) for imp in improvements]
    negative_improvements = [min(0, imp) for imp in improvements]
    
    ax.bar(x, positive_improvements, bottom=base_final_f1, color='#2ca02c', alpha=0.7, label='Positive Improvement')
    ax.bar(x, negative_improvements, bottom=base_final_f1, color='#d62728', alpha=0.7, label='Negative Change')
    
    # Add value labels
    for i, (base, custom, imp) in enumerate(zip(base_final_f1, custom_final_f1, improvements)):
        ax.text(i, custom + 0.02, f'{imp:+.3f}', ha='center', va='bottom', 
               fontsize=9, weight='bold', color='darkgreen' if imp > 0 else 'darkred')
    
    # Formatting
    ax.set_xlabel('Participants', fontweight='bold')
    ax.set_ylabel('F1 Score', fontweight='bold')
    ax.set_title('C. Phase Transition Analysis', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(participants, rotation=45, ha='right')
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, max(custom_final_f1) * 1.1 if custom_final_f1 else 1)

def create_figure_3(participant_data, output_dir):
    """Create the complete Figure 3 with all panels."""
    # Create main figure
    fig = plt.figure(figsize=(16, 20))
    
    # Panel A: Combined curves (top)
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2, fig=fig)
    create_panel_a_combined_curves(ax1, participant_data)
    
    # Panel B: Individual curves (middle) - handled specially
    create_panel_b_individual_curves(fig, participant_data)
    
    # Panel C: Phase analysis (bottom)
    ax3 = plt.subplot2grid((3, 2), (2, 0), colspan=2, fig=fig)
    create_panel_c_phase_analysis(ax3, participant_data)
    
    # Add main title
    fig.suptitle('Two-Phase Training Methodology: Base Training + Personalization', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    # Save figures
    os.makedirs(output_dir, exist_ok=True)
    
    # Vector format for publication
    pdf_path = os.path.join(output_dir, 'figure3_training_curves.pdf')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    
    # Raster format for presentations
    png_path = os.path.join(output_dir, 'figure3_training_curves.png')
    fig.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    
    plt.close()
    
    return pdf_path, png_path

def main():
    parser = argparse.ArgumentParser(description='Generate Figure 3: Training Curves')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing LOPO results with participant subdirectories')
    parser.add_argument('--output_dir', type=str, default='figures',
                       help='Output directory for figures (default: figures/)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.results_dir):
        print(f"❌ Error: Results directory {args.results_dir} does not exist")
        sys.exit(1)
    
    print(f"📈 GENERATING FIGURE 3: TRAINING CURVES")
    print(f"📂 Results directory: {args.results_dir}")
    print(f"🎨 Output directory: {args.output_dir}")
    
    try:
        # Load training data
        print(f"\n📥 Loading participant training data...")
        participant_data = load_participant_training_data(args.results_dir)
        
        if not participant_data:
            print(f"❌ Error: No participant training data found in {args.results_dir}")
            print(f"   Make sure participant subdirectories contain base_metrics.pt and custom_metrics.pt files")
            sys.exit(1)
        
        print(f"   Found training data for {len(participant_data)} participants: {list(participant_data.keys())}")
        
        # Generate figure
        print(f"\n🎨 Creating Figure 3...")
        pdf_path, png_path = create_figure_3(participant_data, args.output_dir)
        
        print(f"\n✅ FIGURE 3 GENERATED SUCCESSFULLY")
        print(f"📄 PDF (publication): {pdf_path}")
        print(f"🖼️  PNG (presentation): {png_path}")
        
        # Summary
        print(f"\n📈 Figure Summary:")
        print(f"   • Participants plotted: {len(participant_data)}")
        print(f"   • Shows two-phase training methodology")
        print(f"   • Individual and combined training curves")
        print(f"   • Phase transition analysis")
        
    except Exception as e:
        print(f"❌ Error generating figure: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()