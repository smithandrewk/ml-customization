"""
This block plots performance from entire experiments directory for something like a hyperparameter sweep.
"""
import os
import json
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, f1_score
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch import nn
from lib.utils_simple import evaluate, compute_loss_and_f1
from lib.models import ConvLayerNorm, Block
import matplotlib.pyplot as plt

data = []
from tqdm import tqdm

experiments_dir = f'./experiments'

for experiment in tqdm(os.listdir(experiments_dir)):
    base_f1s = []
    target_f1s = []
    folds = []

    for run in os.listdir(f'{experiments_dir}/{experiment}'):
        if not os.path.exists(f'{experiments_dir}/{experiment}/{run}/metrics.json'):
            continue
        metrics = json.load(open(f'{experiments_dir}/{experiment}/{run}/metrics.json'))
        losses = json.load(open(f'{experiments_dir}/{experiment}/{run}/losses.json'))
        hyperparameters = json.load(open(f'{experiments_dir}/{experiment}/{run}/hyperparameters.json'))

        custom_model_on_target_val_f1 = metrics['best_base_val_f1'] # temporary naming artifact

        data.append({
            'batch_size': hyperparameters['batch_size'], 
            'fold': run, 
            'best_custom_model_on_target_val_f1': custom_model_on_target_val_f1, 
            'mode': hyperparameters['mode'], 
            'target_data_pct': hyperparameters['target_data_pct'], 
            'best_custom_model_on_target_val_loss': metrics['best_base_val_loss']
        })
        
df = pd.DataFrame(data)

# Create a 2x2 subplot layout
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Model Performance Analysis', fontsize=16)

# Loss plots
sns.lineplot(data=df, x='batch_size', y='best_custom_model_on_target_val_loss', 
             hue='fold', marker='o', ax=axes[0, 0])
axes[0, 0].set_title('Validation Loss by Batch Size (Line Plot)')
axes[0, 0].set_ylabel('Validation Loss')

sns.boxplot(data=df, x='batch_size', y='best_custom_model_on_target_val_loss', ax=axes[0, 1])
axes[0, 1].set_title('Validation Loss Distribution by Batch Size')
axes[0, 1].set_ylabel('Validation Loss')

# F1 plots
sns.lineplot(data=df, x='batch_size', y='best_custom_model_on_target_val_f1', 
             hue='fold', marker='o', ax=axes[1, 0])
axes[1, 0].set_title('Validation F1 by Batch Size (Line Plot)')
axes[1, 0].set_ylabel('Validation F1')

sns.boxplot(data=df, x='batch_size', y='best_custom_model_on_target_val_f1', ax=axes[1, 1])
axes[1, 1].set_title('Validation F1 Distribution by Batch Size')
axes[1, 1].set_ylabel('Validation F1')

plt.tight_layout()
plt.show()

# Target val f1 learning curves
curve_data = []
for experiment in tqdm(os.listdir(experiments_dir)):
    for run in os.listdir(f'{experiments_dir}/{experiment}'):
        if not os.path.exists(f'{experiments_dir}/{experiment}/{run}/losses.json'):
            continue
        losses = json.load(open(f'{experiments_dir}/{experiment}/{run}/losses.json'))

        for epoch, f1 in enumerate(losses['target val f1']):
            curve_data.append({
                'epoch': epoch,
                'target_val_f1': f1,
                'fold': run
            })

curve_df = pd.DataFrame(curve_data)
plt.figure(figsize=(12, 6))
sns.lineplot(data=curve_df, x='epoch', y='target_val_f1', hue='fold')
plt.title('Target Validation F1 Learning Curves')
plt.xlabel('Epoch')
plt.ylabel('Target Val F1')
plt.show()
