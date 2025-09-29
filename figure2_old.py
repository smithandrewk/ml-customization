import os
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import argparse

# Set Nature-style publication parameters
plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 12,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.6,
    'ytick.minor.width': 0.6,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white'
})

# Nature-inspired color palette
colors = {
    'base': '#1f77b4',      # Professional blue
    'target': '#d62728',    # Nature red
    'improve': '#2ca02c',   # Nature green
    'neutral': '#7f7f7f',   # Gray
    'accent': '#ff7f0e'     # Orange accent
}

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate Figure 2 from experiment data')
parser.add_argument('--experiment', type=str, help='Specific experiment directory to plot (e.g., alpha). If not specified, plots all experiments separately.')
args = parser.parse_args()

# Load experiment data
if args.experiment:
    # Plot specific experiment
    if os.path.isdir(f'experiments/{args.experiment}'):
        experiments = [args.experiment]
        print(f"Plotting specific experiment: {args.experiment}")
    else:
        print(f"Error: Experiment directory 'experiments/{args.experiment}' not found")
        print(f"Available experiments: {', '.join(os.listdir('experiments'))}")
        exit(1)
else:
    # Plot all experiments separately
    experiments = [d for d in os.listdir('experiments') if os.path.isdir(f'experiments/{d}')]
    print(f"Plotting each experiment separately: {experiments}")

# Process each experiment separately
for experiment in experiments:
    print(f'\n=== Processing experiment: {experiment} ===')
    data = []
    experiment_dir = f'experiments/{experiment}'

    if not os.path.isdir(experiment_dir):
        continue

    # Get all training run subdirectories
    training_runs = [d for d in os.listdir(experiment_dir)
                    if os.path.isdir(f'{experiment_dir}/{d}') and
                    os.path.exists(f'{experiment_dir}/{d}/hyperparameters.json')]

    if not training_runs:
        print(f"  No valid training runs found in {experiment}")
        continue

    print(f"  Found {len(training_runs)} training runs: {training_runs}")

    # Load data from each training run
    for run_id in training_runs:
        run_dir = f'{experiment_dir}/{run_id}'

        # Check all required files exist
        required_files = ['hyperparameters.json', 'metrics.json', 'losses.json']
        if not all(os.path.exists(f'{run_dir}/{file}') for file in required_files):
            print(f"    Skipping {run_id}: missing required files")
            continue

        try:
            with open(f'{run_dir}/hyperparameters.json', 'r') as f:
                hyperparameters = json.load(f)

            with open(f'{run_dir}/metrics.json', 'r') as f:
                metrics = json.load(f)

            with open(f'{run_dir}/losses.json', 'r') as f:
                losses = json.load(f)

            fold = hyperparameters['fold']
            participants = hyperparameters['participants']
            participants = ['alsaad','anam','asfik','ejaz','iftakhar','tonmoy','unk1','dennis']
            target_participant = participants[fold] if fold < len(participants) else f'fold_{fold}'

            base_f1 = metrics['best_target_val_f1_from_best_base_model']
            target_f1 = metrics['best_target_val_f1']
            improvement = target_f1 - base_f1

            data.append({
                'experiment': f'{experiment}/{run_id}',
                'experiment_name': experiment,
                'run_id': run_id,
                'fold': fold,
                'participant': target_participant,
                'base_f1': base_f1,
                'target_f1': target_f1,
                'improvement': improvement,
                'relative_improvement': improvement / base_f1 * 100,
                'transition_epoch': metrics.get('transition_epoch', 0),
                'losses': losses
            })
            print(f"    Loaded run {run_id}: fold {fold} ({target_participant})")

        except Exception as e:
            print(f"    Error loading {run_id}: {e}")
            continue

    if not data:
        print(f"  No data loaded for experiment {experiment}, skipping...")
        continue

    print(f'  Loaded {len(data)} training runs for {experiment}')
    for d in data:
        print(f'  Fold {d["fold"]} ({d["participant"]}): Base F1: {d["base_f1"]:.4f}, Target F1: {d["target_f1"]:.4f}, Improvement: {d["improvement"]:.4f} ({d["relative_improvement"]:.1f}%)')

    # Create Nature-style figure
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.8))  # Nature single column width
    fig.patch.set_facecolor('white')

    # Panel A: Box plots for F1 score comparison
    ax1 = axes[0, 0]
    base_f1s = [d['base_f1'] for d in data]
    target_f1s = [d['target_f1'] for d in data]

    box_data = [base_f1s, target_f1s]
    bp = ax1.boxplot(box_data, labels=['Base', 'Customized'], patch_artist=True,
                    widths=0.6, medianprops=dict(color='white', linewidth=1.5))
    bp['boxes'][0].set_facecolor(colors['base'])
    bp['boxes'][0].set_alpha(0.8)
    bp['boxes'][1].set_facecolor(colors['target'])
    bp['boxes'][1].set_alpha(0.8)

    # Style whiskers and caps
    for whisker in bp['whiskers']:
        whisker.set_color(colors['neutral'])
        whisker.set_linewidth(1)
    for cap in bp['caps']:
        cap.set_color(colors['neutral'])
        cap.set_linewidth(1)

    # Add statistical test
    t_stat, p_value = stats.ttest_rel(target_f1s, base_f1s)
    if p_value < 0.001:
        p_text = 'P < 0.001'
    else:
        p_text = f'P = {p_value:.3f}'

    ax1.text(0.98, 0.98, p_text, transform=ax1.transAxes, ha='right', va='top',
             fontweight='bold', fontsize=8)

    ax1.set_ylabel('F1 score', fontweight='bold')
    ax1.set_title('a', fontweight='bold', fontsize=12, loc='left', pad=10)
    ax1.grid(True, alpha=0.2, linewidth=0.5)
    ax1.set_ylim(0.0, 1.0)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Panel B: Scatter plot of improvement vs base performance
    ax2 = axes[0, 1]
    improvements = [d['improvement'] for d in data]
    point_colors = [colors['improve'] if imp > 0 else colors['target'] for imp in improvements]

    scatter = ax2.scatter(base_f1s, improvements, c=point_colors, alpha=0.8, s=40,
                         edgecolors='white', linewidths=0.5)

    # Add correlation
    correlation = np.corrcoef(base_f1s, improvements)[0, 1]
    ax2.text(0.02, 0.98, f'r = {correlation:.3f}', transform=ax2.transAxes,
             fontweight='bold', fontsize=8, ha='left', va='top')

    # Add horizontal line at y=0
    ax2.axhline(y=0, color=colors['neutral'], linestyle='-', alpha=0.5, linewidth=1)

    # Add participant labels (smaller, cleaner)
    for i, d in enumerate(data):
        ax2.annotate(f'P{d["fold"]}', (base_f1s[i], improvements[i]),
                    xytext=(3, 3), textcoords='offset points', fontsize=6,
                    alpha=0.7, color=colors['neutral'])

    ax2.set_xlabel('Base F1 score', fontweight='bold')
    ax2.set_ylabel('Improvement (ΔF1)', fontweight='bold')
    ax2.set_title('b', fontweight='bold', fontsize=12, loc='left', pad=10)
    ax2.grid(True, alpha=0.2, linewidth=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Panel C: Bar chart for per-participant results
    ax3 = axes[1, 0]
    participant_labels = [f'P{d["fold"]}' for d in data]
    x_pos = np.arange(len(participant_labels))

    # Create grouped bar chart
    width = 0.35
    bars1 = ax3.bar(x_pos - width/2, base_f1s, width, label='Base',
                    color=colors['base'], alpha=0.8, edgecolor='white', linewidth=0.5)
    bars2 = ax3.bar(x_pos + width/2, target_f1s, width, label='Customized',
                    color=colors['target'], alpha=0.8, edgecolor='white', linewidth=0.5)

    ax3.set_xlabel('Participant', fontweight='bold')
    ax3.set_ylabel('F1 score', fontweight='bold')
    ax3.set_title('c', fontweight='bold', fontsize=12, loc='left', pad=10)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(participant_labels, rotation=45, ha='right', fontsize=7)
    ax3.legend(frameon=False, loc='upper left', fontsize=7)
    ax3.grid(True, alpha=0.2, axis='y', linewidth=0.5)
    ax3.set_ylim(0, 1.0)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # Subplot 4: Training dynamics visualization - aligned by transition
    ax4 = axes[1, 1]

    # Align curves by transition epoch - show full range
    min_transition = min(d['transition_epoch'] for d in data)
    max_epochs_after_transition = max(len(d['losses']['target val loss']) - d['transition_epoch'] for d in data)

    # Go from beginning (0) to end of longest experiment
    pre_transition_epochs = min_transition  # epochs before transition (go to beginning)
    post_transition_epochs = max_epochs_after_transition - 1  # epochs after transition (go to end)
    relative_epochs = range(-pre_transition_epochs, post_transition_epochs + 1)

    # Collect aligned curves
    aligned_curves = []
    for d in data:
        transition = d['transition_epoch']
        target_loss = d['losses']['target val loss']

        aligned_curve = []
        for rel_epoch in relative_epochs:
            abs_epoch = transition + rel_epoch
            if 0 <= abs_epoch < len(target_loss):
                aligned_curve.append(target_loss[abs_epoch])
            else:
                aligned_curve.append(np.nan)
        aligned_curves.append(aligned_curve)

    # Calculate average and std for each relative epoch, but stop when only 1 participant left
    avg_aligned = []
    std_aligned = []
    for i in range(len(relative_epochs)):
        values = [curve[i] for curve in aligned_curves if not np.isnan(curve[i])]
        if len(values) > 2:  # Only include if more than 1 participant
            avg_aligned.append(np.mean(values))
            std_aligned.append(np.std(values))
        else:
            # Stop here - don't include single participant tail
            break

    # Plot individual aligned curves (only up to where we have the average)
    plot_length = len(avg_aligned)
    plot_epochs = relative_epochs[:plot_length]

    # Plot individual curves (subtle)
    for curve in aligned_curves:
        curve_subset = curve[:plot_length]
        valid_epochs = [epoch for epoch, val in zip(plot_epochs, curve_subset) if not np.isnan(val)]
        valid_vals = [val for val in curve_subset if not np.isnan(val)]
        ax4.plot(valid_epochs, valid_vals, color=colors['neutral'], alpha=0.2, linewidth=0.5)

    # Plot average curve
    valid_epochs = plot_epochs
    valid_avg = avg_aligned
    valid_std = std_aligned

    ax4.plot(valid_epochs, valid_avg, color=colors['base'], linewidth=2, label='Mean loss')

    # Add confidence interval
    if len(data) > 2:
        ax4.fill_between(valid_epochs,
                        np.array(valid_avg) - np.array(valid_std),
                        np.array(valid_avg) + np.array(valid_std),
                        alpha=0.2, color=colors['base'])

    # Mark transition point
    ax4.axvline(x=0, color=colors['target'], linestyle='--', linewidth=1.5, alpha=0.8,
               label='Customization')

    ax4.set_xlabel('Epochs relative to transition', fontweight='bold')
    ax4.set_ylabel('Target validation loss', fontweight='bold')
    ax4.set_title('d', fontweight='bold', fontsize=12, loc='left', pad=10)
    ax4.legend(frameon=False, loc='upper right', fontsize=7)
    ax4.grid(True, alpha=0.2, linewidth=0.5)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    # Adjust layout with proper spacing for Nature style
    plt.tight_layout(pad=1.0, h_pad=1.5, w_pad=1.5)

    # Print summary statistics
    print(f'\\n=== Summary Statistics for {experiment} ===')
    print(f'Mean base F1: {np.mean(base_f1s):.4f} ± {np.std(base_f1s):.4f}')
    print(f'Mean customized F1: {np.mean(target_f1s):.4f} ± {np.std(target_f1s):.4f}')
    print(f'Mean improvement: {np.mean(improvements):.4f} ± {np.std(improvements):.4f}')
    print(f'Relative improvement: {np.mean([d["relative_improvement"] for d in data]):.1f}% ± {np.std([d["relative_improvement"] for d in data]):.1f}%')
    print(f'Participants with improvement: {sum(1 for imp in improvements if imp > 0)}/{len(improvements)}')

    # Save figure with experiment-specific filename
    filename = f'figures/figure2_{experiment}.jpg'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'\\nFigure saved as {filename}')

    if args.experiment:
        # Only show plot if specific experiment requested
        plt.show()
    else:
        # Close figure to avoid memory issues when processing multiple experiments
        plt.close()

print("\\nProcessing complete!")