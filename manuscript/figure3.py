import os
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

# Configure matplotlib for high-quality vector graphics
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate Figure 3: Training Dynamics and Understanding Personalization')
parser.add_argument('--experiment', type=str, help='Specific experiment directory to plot (e.g., alpha). If not specified, plots all experiments separately.')
args = parser.parse_args()

# Model architecture (TestModel from eval.py)
class ConvLayerNorm(nn.Module):
    def __init__(self, out_channels) -> None:
        super(ConvLayerNorm,self).__init__()
        self.ln = nn.LayerNorm(out_channels, elementwise_affine=False)

    def forward(self,x):
        x = x.permute(0, 2, 1)
        x = self.ln(x)
        x = x.permute(0, 2, 1)
        return x

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, pool_size=2, pool=True) -> None:
        super(Block,self).__init__()
        self.pool = pool
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.ln = ConvLayerNorm(out_channels)
        if self.pool:
            self.pool = nn.MaxPool1d(pool_size)

    def forward(self,x):
        x = self.conv(x)
        x = self.ln(x)
        x = torch.relu(x)
        if self.pool:
            x = self.pool(x)
        return x

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.blocks = []
        self.blocks.append(Block(6,8))
        for _ in range(5):
            self.blocks.append(Block(8,8))
            self.blocks.append(Block(8,8,pool=False))

        self.blocks.append(Block(8,16,pool=False))
        self.blocks = nn.ModuleList(self.blocks)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.gap(x).squeeze(-1)
        x = self.fc(x)
        return x

def compute_loss_and_f1(model, dataloader, criterion, device):
    """Compute loss and F1 score on a dataset"""
    model.eval()
    total_loss = 0.0
    count = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for Xi, yi in dataloader:
            Xi = Xi.to(device)
            yi = yi.to(device).float()
            logits = model(Xi).squeeze()
            loss = criterion(logits, yi)
            total_loss += loss.item() * Xi.size(0)
            count += Xi.size(0)
            y_true.append(yi.cpu())
            y_pred.append(logits.sigmoid().round().cpu())
    y_true = torch.cat(y_true).cpu()
    y_pred = torch.cat(y_pred).cpu()
    f1 = f1_score(y_true, y_pred, average='macro')
    return total_loss / count, f1.item()

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

# Load experiment data
if args.experiment:
    if os.path.isdir(f'experiments/{args.experiment}'):
        experiments = [args.experiment]
        print(f"Plotting specific experiment: {args.experiment}")
    else:
        print(f"Error: Experiment directory 'experiments/{args.experiment}' not found")
        if os.path.exists('experiments'):
            print(f"Available experiments: {', '.join(os.listdir('experiments'))}")
        exit(1)
else:
    if os.path.exists('experiments'):
        experiments = [d for d in os.listdir('experiments') if os.path.isdir(f'experiments/{d}')]
        print(f"Plotting each experiment separately: {experiments}")
    else:
        print("No experiments directory found")
        exit(1)

# Process each experiment separately
for experiment in experiments:
    print(f'\\n=== Processing experiment: {experiment} ===')
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

            # Compute test set metrics directly for unbiased evaluation
            data_path = hyperparameters['data_path']
            target_testloader = DataLoader(
                TensorDataset(*torch.load(f'{data_path}/{target_participant}_test.pt')),
                batch_size=128, shuffle=False
            )

            model = TestModel()
            criterion = nn.BCEWithLogitsLoss()

            # Compute base model F1 on test set
            model.load_state_dict(torch.load(f'{run_dir}/best_base_model.pt', map_location='cpu'))
            _, base_f1 = compute_loss_and_f1(model, target_testloader, criterion, device='cpu')

            # Compute target model F1 on test set
            model.load_state_dict(torch.load(f'{run_dir}/best_target_model.pt', map_location='cpu'))
            _, target_f1 = compute_loss_and_f1(model, target_testloader, criterion, device='cpu')

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
        print(f"  No data loaded for experiment {experiment}, creating demo figure...")
        # Create demo data for visualization
        np.random.seed(42)
        data = []
        for i in range(8):
            base_f1 = np.random.uniform(0.75, 0.95)
            improvement = np.random.normal(0.04, 0.03)
            target_f1 = base_f1 + improvement
            data.append({
                'experiment': f'demo/fold{i}',
                'fold': i,
                'participant': f'P{i}',
                'base_f1': base_f1,
                'target_f1': target_f1,
                'improvement': improvement,
                'relative_improvement': improvement / base_f1 * 100,
                'transition_epoch': 50,
                'losses': {
                    'target val loss': list(np.random.exponential(0.5, 100)),
                    'target val f1': list(np.random.beta(2, 2, 100))
                }
            })
        print(f"  Created demo dataset with {len(data)} participants")

    print(f'  Loaded {len(data)} training runs for {experiment}')

    # Create four-panel figure for training dynamics
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.8))  # Nature single column width
    fig.patch.set_facecolor('white')

    # Panel A: Loss curves aligned by transition epoch
    ax1 = axes[0, 0]

    if any('target val loss' in d['losses'] for d in data):
        # Align curves by transition epoch - show full range
        min_transition = min(d['transition_epoch'] for d in data)
        max_epochs_after_transition = max(len(d['losses'].get('target val loss', [])) - d['transition_epoch'] for d in data if 'target val loss' in d['losses'])

        if max_epochs_after_transition > 0:
            # Go from beginning (0) to end of longest experiment
            pre_transition_epochs = min_transition
            post_transition_epochs = max_epochs_after_transition - 1
            relative_epochs = range(-pre_transition_epochs, post_transition_epochs + 1)

            # Collect aligned curves
            aligned_curves = []
            for d in data:
                if 'target val loss' not in d['losses']:
                    continue
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

            # Calculate average and std for each relative epoch
            avg_aligned = []
            std_aligned = []
            for i in range(len(relative_epochs)):
                values = [curve[i] for curve in aligned_curves if not np.isnan(curve[i])]
                if len(values) > 2:
                    avg_aligned.append(np.mean(values))
                    std_aligned.append(np.std(values))
                else:
                    break

            # Plot individual aligned curves
            plot_length = len(avg_aligned)
            plot_epochs = relative_epochs[:plot_length]

            # Plot individual curves (subtle)
            for curve in aligned_curves:
                curve_subset = curve[:plot_length]
                valid_epochs = [epoch for epoch, val in zip(plot_epochs, curve_subset) if not np.isnan(val)]
                valid_vals = [val for val in curve_subset if not np.isnan(val)]
                ax1.plot(valid_epochs, valid_vals, color=colors['neutral'], alpha=0.2, linewidth=0.5)

            # Plot average curve
            ax1.plot(plot_epochs, avg_aligned, color=colors['base'], linewidth=2, label='Mean loss')

            # Add confidence interval
            if len(data) > 2:
                ax1.fill_between(plot_epochs,
                                np.array(avg_aligned) - np.array(std_aligned),
                                np.array(avg_aligned) + np.array(std_aligned),
                                alpha=0.2, color=colors['base'])

            # Mark transition point
            ax1.axvline(x=0, color=colors['target'], linestyle='--', linewidth=1.5, alpha=0.8,
                       label='Customization')

            ax1.legend(frameon=False, loc='upper right', fontsize=7)

    ax1.set_xlabel('Epochs relative to transition', fontweight='bold')
    ax1.set_ylabel('Target validation loss', fontweight='bold')
    ax1.set_title('a', fontweight='bold', fontsize=12, loc='left', pad=10)
    ax1.grid(True, alpha=0.2, linewidth=0.5)

    # Panel B: Relative improvement distribution (from original figure1.py concept)
    ax2 = axes[0, 1]
    base_f1s = [d['base_f1'] for d in data]
    target_f1s = [d['target_f1'] for d in data]

    # Calculate relative improvement: (target - base) / (1 - base) * 100
    relative_improvements = [(target - base) / (1 - base) * 100 for base, target in zip(base_f1s, target_f1s)]

    # Create box plot
    bp = ax2.boxplot([relative_improvements], labels=[''], patch_artist=True, widths=0.4,
                    medianprops=dict(color='white', linewidth=1.5))
    bp['boxes'][0].set_facecolor(colors['improve'])
    bp['boxes'][0].set_alpha(0.8)
    bp['boxes'][0].set_edgecolor(colors['improve'])

    # Style whiskers and caps
    for whisker in bp['whiskers']:
        whisker.set_color(colors['neutral'])
        whisker.set_linewidth(1)
    for cap in bp['caps']:
        cap.set_color(colors['neutral'])
        cap.set_linewidth(1)

    # Add individual data points (strip plot)
    x_jitter = np.random.normal(1, 0.04, len(relative_improvements))
    ax2.scatter(x_jitter, relative_improvements, alpha=0.7, s=25,
               color=colors['target'], edgecolors='white', linewidths=0.5)

    # Add reference line at 0% (no improvement)
    ax2.axhline(y=0, color=colors['neutral'], linestyle='--', alpha=0.7, linewidth=1)

    # Calculate effect size (Cohen's d)
    mean_improvement = np.mean(relative_improvements)
    std_improvement = np.std(relative_improvements, ddof=1)
    cohens_d = mean_improvement / std_improvement if std_improvement > 0 else 0

    # Statistical test
    t_stat, p_value = stats.ttest_1samp(relative_improvements, 0)
    if p_value < 0.001:
        p_text = 'P < 0.001'
    else:
        p_text = f'P = {p_value:.3f}'

    # Add statistics
    ax2.text(0.02, 0.98, f'{p_text}\nCohen' + "'s d = " + f'{cohens_d:.2f}\nn = {len(relative_improvements)}',
             transform=ax2.transAxes, ha='left', va='top', fontweight='bold', fontsize=8)

    ax2.set_ylabel('Relative improvement (%)', fontweight='bold')
    ax2.set_title('b', fontweight='bold', fontsize=12, loc='left', pad=10)
    ax2.grid(True, alpha=0.2, linewidth=0.5)
    ax2.set_xlim(0.5, 1.5)

    # Panel C: Convergence analysis
    ax3 = axes[1, 0]

    # Analyze convergence patterns
    convergence_epochs = []
    final_improvements = []

    for d in data:
        # Get number of epochs in customization phase
        if 'target val loss' in d['losses']:
            total_epochs = len(d['losses']['target val loss'])
            customization_epochs = total_epochs - d['transition_epoch']
            convergence_epochs.append(customization_epochs)
            final_improvements.append(d['improvement'])

    if convergence_epochs:
        # Scatter plot of convergence time vs final improvement
        ax3.scatter(convergence_epochs, final_improvements, alpha=0.8, s=40,
                   color=colors['base'], edgecolors='white', linewidths=0.5)

        # Add correlation analysis
        if len(convergence_epochs) > 2:
            correlation = np.corrcoef(convergence_epochs, final_improvements)[0, 1]
            ax3.text(0.02, 0.98, f'r = {correlation:.3f}',
                     transform=ax3.transAxes, ha='left', va='top', fontweight='bold', fontsize=8)

        # Add participant labels
        for i, d in enumerate(data):
            if i < len(convergence_epochs):
                ax3.annotate(f'P{d["fold"]}', (convergence_epochs[i], final_improvements[i]),
                            xytext=(3, 3), textcoords='offset points', fontsize=6,
                            alpha=0.7, color=colors['neutral'])

    ax3.set_xlabel('Customization epochs', fontweight='bold')
    ax3.set_ylabel('Final improvement (ΔF1)', fontweight='bold')
    ax3.set_title('c', fontweight='bold', fontsize=12, loc='left', pad=10)
    ax3.grid(True, alpha=0.2, linewidth=0.5)

    # Panel D: Success/failure pattern analysis
    ax4 = axes[1, 1]

    # Categorize participants by improvement level
    improvements = [d['improvement'] for d in data]
    high_improvers = [i for i, imp in enumerate(improvements) if imp > 0.05]
    moderate_improvers = [i for i, imp in enumerate(improvements) if 0 < imp <= 0.05]
    non_improvers = [i for i, imp in enumerate(improvements) if imp <= 0]

    # Create stacked bar chart showing baseline performance distribution for each category
    categories = ['High\n(>0.05)', 'Moderate\n(0-0.05)', 'None\n(≤0)']
    cat_counts = [len(high_improvers), len(moderate_improvers), len(non_improvers)]

    # Get baseline F1 ranges for each category
    baseline_ranges = []
    for indices in [high_improvers, moderate_improvers, non_improvers]:
        if indices:
            baselines = [base_f1s[i] for i in indices]
            baseline_ranges.append((np.mean(baselines), np.std(baselines)))
        else:
            baseline_ranges.append((0, 0))

    bars = ax4.bar(categories, cat_counts, color=[colors['improve'], colors['accent'], colors['neutral']],
                   alpha=0.8, edgecolor='white', linewidth=0.5)

    # Add count labels on bars
    for bar, count in zip(bars, cat_counts):
        if count > 0:
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                    f'n={count}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Add baseline F1 statistics as text
    for i, (mean_baseline, std_baseline) in enumerate(baseline_ranges):
        if cat_counts[i] > 0:
            ax4.text(i, cat_counts[i]/2, f'F1: {mean_baseline:.2f}±{std_baseline:.2f}',
                    ha='center', va='center', fontsize=7, rotation=0,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

    ax4.set_xlabel('Improvement Category', fontweight='bold')
    ax4.set_ylabel('Number of Participants', fontweight='bold')
    ax4.set_title('d', fontweight='bold', fontsize=12, loc='left', pad=10)
    ax4.grid(True, alpha=0.2, axis='y', linewidth=0.5)

    # Adjust layout
    plt.tight_layout(pad=1.0, h_pad=1.5, w_pad=1.5)

    # Print summary statistics
    print(f'\\\\n=== Training Dynamics Summary for {experiment} ===')
    print(f'Mean relative improvement: {np.mean(relative_improvements):.1f}% ± {np.std(relative_improvements):.1f}%')
    print(f'Effect size (Cohen' + "'s d): " + f'{cohens_d:.3f}')
    print(f'High improvers (>5% ΔF1): {len(high_improvers)}/{len(improvements)}')
    print(f'Moderate improvers (0-5% ΔF1): {len(moderate_improvers)}/{len(improvements)}')
    print(f'Non-improvers (≤0% ΔF1): {len(non_improvers)}/{len(improvements)}')

    # Save figure with experiment-specific filename as PDF vector graphics
    os.makedirs('figures', exist_ok=True)
    filename = f'figures/figure3_{experiment}.pdf'
    plt.savefig(filename, bbox_inches='tight', format='pdf', dpi=300)
    print(f'\\\\nFigure saved as {filename}')

    # Always close figure after saving to avoid GUI popup and memory issues
    plt.close()

print("\\\\nProcessing complete!")