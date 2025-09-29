import os
import json

performances = {}
for experiment in os.listdir('./experiments'):
    base_f1s = []
    target_f1s = []
    folds = []
    for run in os.listdir(f'./experiments/{experiment}'):
        if not os.path.exists(f'./experiments/{experiment}/{run}/metrics.json'):
            continue
        metrics = json.load(open(f'./experiments/{experiment}/{run}/metrics.json'))
        losses = json.load(open(f'./experiments/{experiment}/{run}/losses.json'))

        if 'target_only' in experiment:
            target_model_on_target_val_f1 = losses['target val f1'][metrics['best_base_val_loss_epoch']]
            base_model_on_target_val_f1 = None
        else:
            target_model_on_target_val_f1 = losses['target val f1'][metrics['best_target_val_loss_epoch']]
            base_model_on_target_val_f1 = losses['target val f1'][metrics['best_base_val_loss_epoch']]

        base_f1s.append(base_model_on_target_val_f1)
        target_f1s.append(target_model_on_target_val_f1)
        folds.append(run)
    performances[experiment] = (base_f1s, target_f1s, folds)

# boxplot all performances
import pandas as pd
data = []
for experiment, (base_f1s, target_f1s, folds) in performances.items():
    for base_f1, target_f1, fold in zip(base_f1s, target_f1s, folds):
        data.append({'experiment': experiment, 'model': 'base', 'f1': base_f1, 'fold': fold})
        data.append({'experiment': experiment, 'model': 'target', 'f1': target_f1, 'fold': fold})
df = pd.DataFrame(data)

import seaborn as sns

# Prepare data: exclude base model, rename strategies, extract percentages
df_b = df.copy()
df_b = df_b[~(df_b['model'] == 'base')]
df_b.loc[df_b['experiment'].str.contains('full'),'model'] = 'Full Fine-Tuning'
df_b.loc[df_b['experiment'].str.contains('target'),'model'] = 'Target-Only'
df_b['experiment'] = df_b['experiment'].str.split('_').str[-3].str.replace('pct','').astype(float)
df_b.rename(columns={'experiment':'Target Training Data (%)'}, inplace=True)

# Create figure with aggregate plot showing all folds
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 6))

# Plot with confidence intervals across folds
sns.lineplot(x='Target Training Data (%)', y='f1', hue='model',
             marker='o', markersize=8, linewidth=2.5,
             data=df_b, ax=ax, errorbar='sd')

# Highlight the crossover region
ax.axhline(y=df_b[df_b['model']=='Target-Only']['f1'].max(),
           color='gray', linestyle='--', alpha=0.3, linewidth=1)

ax.set_ylabel('F1 Score', fontsize=12)
ax.set_xlabel('Target Training Data (%)', fontsize=12)
ax.set_title('Full Fine-Tuning with Limited Data Outperforms\nTarget-Only Training with More Data',
             fontsize=13, fontweight='bold', pad=15)
ax.legend(title='Training Strategy', fontsize=11, title_fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim([df_b['f1'].min() - 0.02, df_b['f1'].max() + 0.02])

plt.tight_layout()
plt.savefig('figure_idea.png', dpi=300, bbox_inches='tight')
plt.close()