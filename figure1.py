import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from scipy import stats
import argparse
import pandas as pd

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate Figure 1: Study Design and Methodology')
parser.add_argument('--experiment', type=str, help='Specific experiment directory to plot')
args = parser.parse_args()

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
    'accent': '#ff7f0e',    # Orange accent
    'light_blue': '#add8e6',
    'light_green': '#90ee90',
    'light_red': '#ffb6c1'
}

# Create four-panel figure
fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.8))  # Nature single column width
fig.patch.set_facecolor('white')

# Panel A: Study Workflow Diagram
ax1 = axes[0, 0]

# Create workflow steps
steps = ['Population\nData', 'Base Model\nTraining', 'Target\nPersonalization', 'Evaluation']
y_positions = [0.8, 0.5, 0.5, 0.2]
x_positions = [0.2, 0.2, 0.8, 0.8]

# Draw boxes for each step
for i, (step, x, y) in enumerate(zip(steps, x_positions, y_positions)):
    if i == 0:
        color = colors['neutral']
    elif i == 1:
        color = colors['base']
    elif i == 2:
        color = colors['target']
    else:
        color = colors['improve']

    box = mpatches.FancyBboxPatch((x-0.1, y-0.08), 0.2, 0.16,
                                  boxstyle="round,pad=0.02",
                                  facecolor=color, alpha=0.3,
                                  edgecolor=color, linewidth=1.5)
    ax1.add_patch(box)
    ax1.text(x, y, step, ha='center', va='center', fontweight='bold', fontsize=8)

# Draw arrows
arrow_props = dict(arrowstyle='->', lw=1.5, color=colors['neutral'])
# Population to Base Model
ax1.annotate('', xy=(0.2, 0.42), xytext=(0.2, 0.72), arrowprops=arrow_props)
# Base Model to Personalization
ax1.annotate('', xy=(0.7, 0.5), xytext=(0.3, 0.5), arrowprops=arrow_props)
# Personalization to Evaluation
ax1.annotate('', xy=(0.8, 0.28), xytext=(0.8, 0.42), arrowprops=arrow_props)

# Add labels
ax1.text(0.15, 0.57, 'Leave-one-out\nCV', ha='center', va='center', fontsize=7, style='italic')
ax1.text(0.5, 0.53, 'Fine-tuning', ha='center', va='center', fontsize=7, style='italic')
ax1.text(0.85, 0.35, 'F1 score', ha='center', va='center', fontsize=7, style='italic')

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_title('a', fontweight='bold', fontsize=12, loc='left', pad=10)
ax1.axis('off')

# Panel B: Dataset Characteristics
ax2 = axes[0, 1]

# Simulated dataset statistics (replace with real data when available)
participants = ['P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7']
smoking_sessions = [23, 18, 31, 27, 20, 25, 22, 19]  # Example data
non_smoking_sessions = [145, 132, 178, 154, 138, 162, 149, 143]

x_pos = np.arange(len(participants))
width = 0.35

bars1 = ax2.bar(x_pos - width/2, smoking_sessions, width,
                label='Smoking', color=colors['target'], alpha=0.8)
bars2 = ax2.bar(x_pos + width/2, non_smoking_sessions, width,
                label='Non-smoking', color=colors['base'], alpha=0.8)

ax2.set_xlabel('Participant', fontweight='bold')
ax2.set_ylabel('Number of Sessions', fontweight='bold')
ax2.set_title('b', fontweight='bold', fontsize=12, loc='left', pad=10)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(participants)
ax2.legend(frameon=False, loc='upper right', fontsize=7)
ax2.grid(True, alpha=0.2, axis='y', linewidth=0.5)

# Panel C: Model Architecture Diagram
ax3 = axes[1, 0]

# Create simplified CNN architecture visualization
layers = ['Input\n(3000×6)', 'Conv1D\n16 filters', 'Conv1D\n32 filters', 'Conv1D\n64 filters', 'Global\nAvgPool', 'Dense\n(1)']
y_pos = 0.5
x_positions = np.linspace(0.1, 0.9, len(layers))
widths = [0.12, 0.12, 0.12, 0.12, 0.08, 0.08]

for i, (layer, x, w) in enumerate(zip(layers, x_positions, widths)):
    if i == 0:
        color = colors['light_blue']
    elif i < 4:
        color = colors['light_green']
    elif i == 4:
        color = colors['light_red']
    else:
        color = colors['accent']

    box = mpatches.FancyBboxPatch((x-w/2, y_pos-0.1), w, 0.2,
                                  boxstyle="round,pad=0.01",
                                  facecolor=color, alpha=0.6,
                                  edgecolor='black', linewidth=0.8)
    ax3.add_patch(box)
    ax3.text(x, y_pos, layer, ha='center', va='center', fontweight='bold', fontsize=7)

# Draw connections
for i in range(len(x_positions)-1):
    ax3.annotate('', xy=(x_positions[i+1]-widths[i+1]/2, y_pos),
                 xytext=(x_positions[i]+widths[i]/2, y_pos),
                 arrowprops=dict(arrowstyle='->', lw=1, color='black'))

ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.set_title('c', fontweight='bold', fontsize=12, loc='left', pad=10)
ax3.axis('off')

# Panel D: Sample Accelerometer Data
ax4 = axes[1, 1]

# Generate realistic-looking accelerometer data
np.random.seed(42)
time_points = np.linspace(0, 60, 3000)  # 60 seconds at 50Hz

# Simulate different phases: non-smoking, smoking gesture, smoking
base_signal = np.sin(0.1 * time_points) + 0.1 * np.random.randn(3000)
smoking_period = (time_points > 20) & (time_points < 25)
gesture_enhancement = 2 * np.sin(2 * time_points[smoking_period]) * np.exp(-(time_points[smoking_period]-22.5)**2/2)

# Add smoking gestures
accelerometer_x = base_signal.copy()
accelerometer_x[smoking_period] += gesture_enhancement

# Plot accelerometer trace
ax4.plot(time_points, accelerometer_x, color=colors['neutral'], linewidth=0.8, alpha=0.8)

# Highlight smoking bout
smoking_start, smoking_end = 20, 25
ax4.axvspan(smoking_start, smoking_end, alpha=0.3, color=colors['target'], label='Smoking bout')

# Add annotation
ax4.annotate('Smoking\ngestures', xy=(22.5, max(accelerometer_x[smoking_period])),
            xytext=(35, 1.5), fontsize=7,
            arrowprops=dict(arrowstyle='->', color=colors['target'], lw=1))

ax4.set_xlabel('Time (seconds)', fontweight='bold')
ax4.set_ylabel('Acceleration (g)', fontweight='bold')
ax4.set_title('d', fontweight='bold', fontsize=12, loc='left', pad=10)
ax4.grid(True, alpha=0.2, linewidth=0.5)
ax4.legend(frameon=False, loc='upper right', fontsize=7)
ax4.set_xlim(0, 60)

# Adjust layout
plt.tight_layout(pad=1.0, h_pad=1.5, w_pad=1.5)

# Save figure
os.makedirs('figures', exist_ok=True)
filename = 'figures/figure1.jpg'
plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f'Figure 1 saved as {filename}')

if args.experiment:
    plt.show()
else:
    plt.close()

print("Figure 1 generation complete!")