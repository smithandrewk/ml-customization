import os
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

# Configure matplotlib for high-quality vector graphics
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate Figure 4: Transfer Learning Data Efficiency Analysis')
parser.add_argument('--experiment-prefix', type=str, default='b128_aug_patience50',
                   help='Experiment name prefix to filter experiments')
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
    'full': '#1f77b4',      # Professional blue for full fine-tuning
    'target': '#d62728',    # Nature red for target-only
    'neutral': '#7f7f7f',   # Gray
}

print(f'=== Generating Figure 4: Transfer Learning Data Efficiency ===')

# Load experiment data
performances = {}
experiment_dir = './experiments'

if not os.path.exists(experiment_dir):
    print(f"Error: experiments directory not found")
    exit(1)

for experiment in os.listdir(experiment_dir):
    # Filter by experiment prefix if specified
    if args.experiment_prefix and not experiment.startswith(args.experiment_prefix):
        continue

    base_f1s = []
    target_f1s = []
    folds = []

    exp_path = f'{experiment_dir}/{experiment}'
    if not os.path.isdir(exp_path):
        continue

    for run in os.listdir(exp_path):
        run_path = f'{exp_path}/{run}'
        metrics_file = f'{run_path}/metrics.json'
        losses_file = f'{run_path}/losses.json'
        hyperparameters_file = f'{run_path}/hyperparameters.json'

        if not os.path.exists(metrics_file) or not os.path.exists(losses_file) or not os.path.exists(hyperparameters_file):
            continue

        metrics = json.load(open(metrics_file))
        losses = json.load(open(losses_file))
        hyperparameters = json.load(open(hyperparameters_file))

        # Compute test set metrics directly for unbiased evaluation
        fold = hyperparameters['fold']
        participants = hyperparameters.get('participants', ['alsaad','anam','asfik','ejaz','iftakhar','tonmoy','unk1','dennis'])
        target_participant = participants[fold] if fold < len(participants) else f'fold_{fold}'
        data_path = hyperparameters['data_path']

        target_testloader = DataLoader(
            TensorDataset(*torch.load(f'{data_path}/{target_participant}_test.pt')),
            batch_size=128, shuffle=False
        )

        model = TestModel()
        criterion = nn.BCEWithLogitsLoss()

        if 'target_only' in experiment:
            # For target-only, the base model IS the target model (trained only on target data)
            base_model_on_target_test_f1 = None
            model.load_state_dict(torch.load(f'{run_path}/best_base_model.pt', map_location='cpu'))
            _, target_model_on_target_test_f1 = compute_loss_and_f1(model, target_testloader, criterion, device='cpu')
        else:
            # For transfer learning (full fine-tuning), compute test metrics
            model.load_state_dict(torch.load(f'{run_path}/best_base_model.pt', map_location='cpu'))
            _, base_model_on_target_test_f1 = compute_loss_and_f1(model, target_testloader, criterion, device='cpu')

            model.load_state_dict(torch.load(f'{run_path}/best_target_model.pt', map_location='cpu'))
            _, target_model_on_target_test_f1 = compute_loss_and_f1(model, target_testloader, criterion, device='cpu')

        base_f1s.append(base_model_on_target_test_f1)
        target_f1s.append(target_model_on_target_test_f1)
        folds.append(run)

    performances[experiment] = (base_f1s, target_f1s, folds)

print(f'Loaded {len(performances)} experiments')

# Prepare data for visualization
data = []
for experiment, (base_f1s, target_f1s, folds) in performances.items():
    for base_f1, target_f1, fold in zip(base_f1s, target_f1s, folds):
        data.append({'experiment': experiment, 'model': 'base', 'f1': base_f1, 'fold': fold})
        data.append({'experiment': experiment, 'model': 'target', 'f1': target_f1, 'fold': fold})

if not data:
    print("Warning: No experiment data found. Please run experiments first.")
    exit(1)

df = pd.DataFrame(data)

# Prepare data: exclude base model, rename strategies, extract percentages
df_plot = df.copy()
df_plot = df_plot[~(df_plot['model'] == 'base')]
df_plot.loc[df_plot['experiment'].str.contains('full'),'model'] = 'Full Fine-Tuning'
df_plot.loc[df_plot['experiment'].str.contains('target'),'model'] = 'Target-Only'

# Extract percentage from experiment name (e.g., "pct0.05" -> 5.0)
df_plot['experiment'] = df_plot['experiment'].str.split('_').str[-3].str.replace('pct','').astype(float) * 100
df_plot.rename(columns={'experiment':'Target Training Data (%)'}, inplace=True)

print(f'Plotting {len(df_plot)} data points')
print(f'Strategies: {df_plot["model"].unique()}')
print(f'Data percentages: {sorted(df_plot["Target Training Data (%)"].unique())}')

# Create figure
fig, ax = plt.subplots(figsize=(7.2, 4.5))

# Plot with confidence intervals across folds
sns.lineplot(x='Target Training Data (%)', y='f1', hue='model',
             marker='o', markersize=8, linewidth=2.5,
             data=df_plot, ax=ax, errorbar='sd',
             palette={'Full Fine-Tuning': colors['full'], 'Target-Only': colors['target']})

# Highlight the crossover region with a subtle reference line
max_target_f1 = df_plot[df_plot['model']=='Target-Only']['f1'].max()
ax.axhline(y=max_target_f1, color=colors['neutral'], linestyle='--',
           alpha=0.3, linewidth=1, zorder=1)

# Add annotation to highlight key finding
min_full_ft = df_plot[df_plot['model']=='Full Fine-Tuning'].groupby('Target Training Data (%)')['f1'].mean().min()
if min_full_ft > max_target_f1:
    ax.text(0.98, 0.02,
            'Full fine-tuning with limited data\noutperforms target-only with full data',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=8, style='italic',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor=colors['neutral']))

ax.set_ylabel('F1 Score', fontweight='bold')
ax.set_xlabel('Target Training Data (%)', fontweight='bold')
ax.set_title('Transfer Learning Data Efficiency: Population Knowledge Enables\nSuperior Performance with Limited Target Data',
             fontsize=11, fontweight='bold', pad=15)
ax.legend(title='Training Strategy', fontsize=8, title_fontsize=9, frameon=False, loc='lower right')
ax.grid(True, alpha=0.3, linewidth=0.5)

# Set appropriate y-axis limits
y_min = df_plot['f1'].min() - 0.02
y_max = df_plot['f1'].max() + 0.02
ax.set_ylim([max(0, y_min), min(1, y_max)])

# Set x-axis to log scale if there's a wide range
data_percentages = sorted(df_plot['Target Training Data (%)'].unique())
if max(data_percentages) / min(data_percentages) > 10:
    ax.set_xscale('log')
    ax.set_xticks(data_percentages)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

plt.tight_layout()

# Save figure as PDF vector graphics
os.makedirs('figures', exist_ok=True)
filename = 'figures/figure4.pdf'
plt.savefig(filename, bbox_inches='tight', format='pdf', dpi=300)
print(f'\nFigure 4 saved as {filename}')

# Always close the figure after saving to avoid GUI popup
plt.close()

# Print summary statistics
print('\n=== Summary Statistics ===')
for strategy in ['Full Fine-Tuning', 'Target-Only']:
    print(f'\n{strategy}:')
    strategy_data = df_plot[df_plot['model'] == strategy]
    for pct in sorted(strategy_data['Target Training Data (%)'].unique()):
        pct_data = strategy_data[strategy_data['Target Training Data (%)'] == pct]['f1']
        print(f'  {pct:.1f}% data: F1 = {pct_data.mean():.4f} ± {pct_data.std():.4f} (n={len(pct_data)})')

print("\nFigure 4 generation complete!")