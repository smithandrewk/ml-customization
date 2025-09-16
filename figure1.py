import os
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

# Load all experiment data
experiments = os.listdir('experiments')
data = []

for experiment in experiments:
    if not os.path.isdir(f'experiments/{experiment}'):
        continue

    if not os.path.exists(f'experiments/{experiment}/hyperparameters.json'):
        continue

    if not os.path.exists(f'experiments/{experiment}/metrics.json'):
        continue

    with open(f'experiments/{experiment}/hyperparameters.json', 'r') as f:
        hyperparameters = json.load(f)

    with open(f'experiments/{experiment}/metrics.json', 'r') as f:
        metrics = json.load(f)

    with open(f'experiments/{experiment}/losses.json', 'r') as f:
        losses = json.load(f)

    fold = hyperparameters['fold']
    participants = hyperparameters['participants']
    participants = ['alsaad','anam','asfik','ejaz','iftakhar','tonmoy','unk1','dennis']
    target_participant = participants[fold] if fold < len(participants) else f'fold_{fold}'

    base_f1 = metrics['best_target_val_f1_from_best_base_model']
    target_f1 = metrics['best_target_val_f1']
    improvement = target_f1 - base_f1

    data.append({
        'experiment': experiment,
        'fold': fold,
        'participant': target_participant,
        'base_f1': base_f1,
        'target_f1': target_f1,
        'improvement': improvement,
        'relative_improvement': improvement / base_f1 * 100,
        'transition_epoch': metrics.get('transition_epoch', 0),
        'losses': losses
    })

print(f'Loaded {len(data)} experiments')
for d in data:
    print(f'Fold {d["fold"]} ({d["participant"]}): Base F1: {d["base_f1"]:.4f}, Target F1: {d["target_f1"]:.4f}, Improvement: {d["improvement"]:.4f} ({d["relative_improvement"]:.1f}%)')

# Set Nature-style publication parameters
plt.rcParams.update({
    'font.family': 'Arial',
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

# Create two-panel figure
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))  # Nature single column width
fig.patch.set_facecolor('white')

# Panel A: Enhanced box plot with individual points and effect size
ax1 = axes[0]
base_f1s = [d['base_f1'] for d in data]
target_f1s = [d['target_f1'] for d in data]

# Calculate relative improvement: (target - base) / (1 - base) * 100
relative_improvements = [(target - base) / (1 - base) * 100 for base, target in zip(base_f1s, target_f1s)]

# Create box plot
bp = ax1.boxplot([relative_improvements], labels=[''], patch_artist=True, widths=0.4,
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
x_jitter = np.random.normal(1, 0.04, len(relative_improvements))  # Add slight jitter
ax1.scatter(x_jitter, relative_improvements, alpha=0.7, s=25,
           color=colors['target'], edgecolors='white', linewidths=0.5)

# Add reference line at 0% (no improvement)
ax1.axhline(y=0, color=colors['neutral'], linestyle='--', alpha=0.7, linewidth=1)

# Calculate effect size (Cohen's d)
mean_improvement = np.mean(relative_improvements)
std_improvement = np.std(relative_improvements, ddof=1)
cohens_d = mean_improvement / std_improvement

# Statistical test
t_stat, p_value = stats.ttest_1samp(relative_improvements, 0)
if p_value < 0.001:
    p_text = 'P < 0.001'
else:
    p_text = f'P = {p_value:.3f}'

# Add statistics
ax1.text(0.02, 0.98, f'{p_text}\nCohen\'s d = {cohens_d:.2f}\nn = {len(relative_improvements)}',
         transform=ax1.transAxes, ha='left', va='top', fontweight='bold', fontsize=8)

ax1.set_ylabel('Relative improvement (%)', fontweight='bold')
ax1.set_title('a', fontweight='bold', fontsize=12, loc='left', pad=10)
ax1.grid(True, alpha=0.2, linewidth=0.5)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_xlim(0.5, 1.5)

# Panel B: Scatter plot showing relative improvement vs baseline performance
ax2 = axes[1]

# Color points based on improvement direction
point_colors = [colors['improve'] if imp > 0 else colors['target'] for imp in relative_improvements]

# Create scatter plot
scatter = ax2.scatter(base_f1s, relative_improvements, c=point_colors, alpha=0.8, s=40,
                     edgecolors='white', linewidths=0.5)

# Add correlation analysis
correlation = np.corrcoef(base_f1s, relative_improvements)[0, 1]
r_squared = correlation ** 2

# Add trend line if correlation is significant
from scipy.stats import pearsonr
r, p_corr = pearsonr(base_f1s, relative_improvements)
if p_corr < 0.05:
    # Add trend line
    z = np.polyfit(base_f1s, relative_improvements, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(base_f1s), max(base_f1s), 100)
    ax2.plot(x_trend, p(x_trend), color=colors['neutral'], linestyle='-', alpha=0.8, linewidth=1)

# Add horizontal line at y=0
ax2.axhline(y=0, color=colors['neutral'], linestyle='--', alpha=0.7, linewidth=1)

# Add participant labels
for i, d in enumerate(data):
    ax2.annotate(f'P{d["fold"]}', (base_f1s[i], relative_improvements[i]),
                xytext=(3, 3), textcoords='offset points', fontsize=6,
                alpha=0.7, color=colors['neutral'])

# Add correlation statistics
if p_corr < 0.001:
    p_corr_text = 'P < 0.001'
else:
    p_corr_text = f'P = {p_corr:.3f}'

ax2.text(0.02, 0.98, f'r = {correlation:.3f}\n{p_corr_text}',
         transform=ax2.transAxes, ha='left', va='top', fontweight='bold', fontsize=8)

ax2.set_xlabel('Baseline F1 score', fontweight='bold')
ax2.set_ylabel('Relative improvement (%)', fontweight='bold')
ax2.set_title('b', fontweight='bold', fontsize=12, loc='left', pad=10)
ax2.grid(True, alpha=0.2, linewidth=0.5)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Adjust layout with proper spacing
plt.tight_layout(pad=1.0, w_pad=2.0)

# Print summary statistics
absolute_improvements = [d['improvement'] for d in data]
median_improvement = np.median(relative_improvements)
print(f'\n=== Summary Statistics ===')
print(f'Mean base F1: {np.mean(base_f1s):.4f} ± {np.std(base_f1s):.4f}')
print(f'Mean customized F1: {np.mean(target_f1s):.4f} ± {np.std(target_f1s):.4f}')
print(f'Mean absolute improvement: {np.mean(absolute_improvements):.4f} ± {np.std(absolute_improvements):.4f}')
print(f'Mean relative improvement: {np.mean(relative_improvements):.1f}% ± {np.std(relative_improvements):.1f}%')
print(f'Median relative improvement: {median_improvement:.1f}%')
print(f'Cohen\'s d: {cohens_d:.3f}')
print(f'Correlation with baseline: r = {correlation:.3f} (p = {p_corr:.3f})')
print(f'Participants with improvement: {sum(1 for imp in relative_improvements if imp > 0)}/{len(relative_improvements)}')

# Save figure
plt.savefig('figures/figure1.jpg', dpi=300, bbox_inches='tight', facecolor='white')
print(f'\nFigure saved as figures/figure1.jpg')

plt.show()
