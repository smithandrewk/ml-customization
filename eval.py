# %%
"""
This block plots performance from entire experiments directory for something like a hyperparameter sweep.
"""
import os
import json
import pandas as pd
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch import nn
import matplotlib.pyplot as plt
from tqdm import tqdm

data = []

experiments_dir = f'./experiments'

for experiment in tqdm(os.listdir(experiments_dir)):
    for run in os.listdir(f'{experiments_dir}/{experiment}'):
        if not os.path.exists(f'{experiments_dir}/{experiment}/{run}/metrics.json'):
            continue
        
        metrics = json.load(open(f'{experiments_dir}/{experiment}/{run}/metrics.json'))
        losses = json.load(open(f'{experiments_dir}/{experiment}/{run}/losses.json'))
        hyperparameters = json.load(open(f'{experiments_dir}/{experiment}/{run}/hyperparameters.json'))

        data.append({
            'hyperparameter_hash': experiment.split('_')[-1],
            'batch_size': hyperparameters['batch_size'], 
            'fold': int(run.split('_')[0].replace('fold','')), 
            'target_data_pct': float(hyperparameters['target_data_pct']), 
            'n_base_participants': int(hyperparameters['n_base_participants']),
            'mode': hyperparameters['mode'],
            'best_target_model_target_val_f1': metrics['best_target_model_target_val_f1'], 
            'best_target_model_target_test_f1': metrics['best_target_model_target_test_f1'],
            'best_base_model_target_val_f1': metrics['best_base_model_target_val_f1'] if 'best_base_model_target_val_f1' in metrics else None,
            'best_base_model_target_test_f1': metrics['best_base_model_target_test_f1'] if 'best_base_model_target_test_f1' in metrics else None,
            'best_target_model_target_val_loss': metrics['best_target_model_target_val_loss'] if 'best_target_model_target_val_loss' in metrics else None,
            'best_target_model_target_test_loss': metrics['best_target_model_target_test_loss'] if 'best_target_model_target_test_loss' in metrics else None,
            'best_base_model_target_val_loss': metrics['best_base_model_target_val_loss'] if 'best_base_model_target_val_loss' in metrics else None,
            'best_base_model_target_test_loss': metrics['best_base_model_target_test_loss'] if 'best_base_model_target_test_loss' in metrics else None,
        })
        
df = pd.DataFrame(data)

# %%
print(df['hyperparameter_hash'].value_counts())
print(df['mode'].value_counts())
print(df['target_data_pct'].value_counts())
print(df['batch_size'].value_counts())
print(df['n_base_participants'].value_counts())

# %%
df_plot = df.copy()
sns.boxplot(data=df_plot, x='fold', y='best_target_model_target_val_f1', hue='mode')

# %%
df_plot = df.copy()
# df_plot = df_plot[df_plot['mode'] == 'full_fine_tuning']
df_plot = df_plot[df_plot['target_data_pct'] == 1]
# df_plot = df_plot[df_plot['n_base_participants'] == 6]
df_plot = df_plot.melt(value_vars=['best_base_model_target_test_f1','best_target_model_target_test_f1'],id_vars=['fold','batch_size','target_data_pct','n_base_participants','mode'], var_name='metric', value_name='value')
sns.boxplot(data=df_plot, x='metric', y='value', hue='mode')

# %%
df_plot = df.copy()
df_plot = df_plot[df_plot['mode'] == 'target_only_fine_tuning']
# df_plot = df_plot[df_plot['target_data_pct'] == 1]
# df_plot = df_plot[df_plot['n_base_participants'] == 6]
df_plot = df_plot.melt(value_vars=['best_base_model_target_test_f1','best_target_model_target_test_f1'],id_vars=['fold','batch_size','target_data_pct','n_base_participants','mode'], var_name='metric', value_name='value')
df_plot
sns.boxplot(data=df_plot, x='fold', y='value', hue='metric')

# %%


# %%
sns.boxplot(data=df, x='mode', y='best_target_model_target_test_f1')

# %%
sns.boxplot(data=df, x='mode', y='best_target_model_target_test_f1', hue='target_data_pct')

# %%
fig,ax = plt.subplots(2,2, figsize=(12,10))
sns.boxplot(data=df, x='target_data_pct', y='best_target_model_target_val_loss', ax=ax[0,0])
ax[0,0].set_yscale('log')
sns.boxplot(data=df, x='target_data_pct', y='best_target_model_target_val_f1', ax=ax[1,0])
sns.boxplot(data=df, x='batch_size', y='best_target_model_target_val_f1', hue='target_data_pct', ax=ax[0,1])
sns.boxplot(data=df, x='batch_size', y='best_target_model_target_val_loss', hue='target_data_pct', ax=ax[1,1])

# %%
df_plot = df.copy()
df_plot = df_plot[df_plot['mode'] == 'target_only_fine_tuning']
# df_plot = df_plot[df_plot['target_data_pct'] == 1]
sns.boxplot(data=df_plot, x='n_base_participants', y='best_target_model_target_test_f1')

# %%
df_plot = df.copy()
df_plot = df_plot[df_plot['mode'] == 'full_fine_tuning']
# df_plot = df_plot[df_plot['target_data_pct'] == 1]
sns.boxplot(data=df_plot, x='n_base_participants', y='best_target_model_target_test_f1')

# %%
df_plot = df.copy()
# df_plot = df_plot[df_plot['mode'] == 'target_only_fine_tuning']
sns.boxplot(data=df_plot, x='fold', y='best_target_model_target_test_f1', hue='mode')

# %%


# %%
best_model_hash = df.groupby('hyperparameter_hash')['absolute_improvement'].mean().sort_values(ascending=False).keys()[0]
print(best_model_hash)
df_metrics = df[df['hyperparameter_hash'] == best_model_hash].copy()
# Compute Metrics For Paper
display(df_metrics)
df_metrics = df_metrics[['fold','best_base_model_target_test_f1','best_target_model_target_test_f1']]
df_metrics.sort_values('fold', inplace=True)
# Add Absolute Improvement
df_metrics['absolute_improvement'] = df_metrics['best_target_model_target_test_f1'] - df_metrics['best_base_model_target_test_f1']
# Add Relative Improvement
df_metrics['relative_improvement'] = df_metrics['absolute_improvement'] / df_metrics['best_base_model_target_test_f1']
# Add Room For Improvement Metric
df_metrics['room_for_improvement'] = (1 - df_metrics['best_base_model_target_test_f1'])
df_metrics['room_for_improvement_pct'] = df_metrics['absolute_improvement'] / df_metrics['room_for_improvement']

# %%
from lib.models import TestModel
from lib.train_utils import compute_loss_and_f1,evaluate
from sklearn.metrics import classification_report
from lib.train_utils import random_subsample
device = 'cpu'

# Load Model From Best Model Hash
# You can change this to load a different model if you want to inspect it
best_model_dir = [d for d in os.listdir(experiments_dir) if best_model_hash in d][0]
print(f'Loading model from {best_model_dir}')

run_dirs = os.listdir(f'{experiments_dir}/{best_model_dir}')
print(f'Runs: {run_dirs}')

df_metrics['best_target_model_target_test_precision'] = 0.0
df_metrics['best_target_model_target_test_recall'] = 0.0

for run_dir in run_dirs:
    print(f'Loading model from {run_dir}')
    best_base_model_path = f'{experiments_dir}/{best_model_dir}/{run_dir}/best_base_model.pt'

    criterion = nn.BCEWithLogitsLoss()
    hyperparameters = json.load(open(f'{experiments_dir}/{best_model_dir}/{run_dir}/hyperparameters.json'))
    target_participant = hyperparameters['target_participant']
    data_path = hyperparameters['data_path']
    batch_size = hyperparameters['batch_size']
    fold = hyperparameters['fold']
    print(f'Loading data from {data_path} for target participant {target_participant}')
    target_train_dataset = TensorDataset(*torch.load(f'{data_path}/{target_participant}_train.pt'))
    target_val_dataset = TensorDataset(*torch.load(f'{data_path}/{target_participant}_val.pt'))
    target_test_dataset = TensorDataset(*torch.load(f'{data_path}/{target_participant}_test.pt'))

    # Subsample target training data if specified
    target_data_pct = hyperparameters['target_data_pct']
    if target_data_pct < 1.0:
        print(f'Target train dataset size: {len(target_train_dataset)}')
        target_train_dataset = random_subsample(target_train_dataset, target_data_pct)
        print(f'Target train dataset size: {len(target_train_dataset)}')

    print(f'Target val dataset size: {len(target_val_dataset)}')
    target_val_dataset = random_subsample(target_val_dataset, .1)
    print(f'Target val dataset size: {len(target_val_dataset)}')

    target_trainloader = DataLoader(target_train_dataset, batch_size=batch_size, shuffle=True)
    target_valloader = DataLoader(target_val_dataset, batch_size=batch_size, shuffle=False)
    target_testloader = DataLoader(target_test_dataset, batch_size=batch_size, shuffle=False)

    model = TestModel()
    print(f'Loading base model from {best_base_model_path}')
    model.load_state_dict(torch.load(best_base_model_path, map_location='cpu'))
    model.to(device)

    test_loss, test_f1 = compute_loss_and_f1(model, target_testloader, criterion, device=device)
    y_true,y_pred = evaluate(model, target_testloader, device=device)

    report = classification_report(y_true, y_pred, output_dict=True)
    precision = report['macro avg']['precision']
    recall = report['macro avg']['recall']

    df_metrics.loc[df_metrics['fold'] == fold,'best_target_model_target_test_precision'] = precision
    df_metrics.loc[df_metrics['fold'] == fold,'best_target_model_target_test_recall'] = recall

# %%
# Add Mean and Std and Median
df_metrics.loc['median'] = df_metrics.median(numeric_only=True)
df_metrics.loc['mean'] = df_metrics.mean(numeric_only=True)
df_metrics.loc['std'] = df_metrics.std(numeric_only=True)

display(df_metrics)

df_metrics.drop(['median','mean','std'], inplace=True)
df_metrics_melted = df_metrics.reset_index().melt(id_vars=['fold'], value_vars=['best_base_model_target_test_f1','best_target_model_target_test_f1','best_target_model_target_test_precision','best_target_model_target_test_recall','absolute_improvement','relative_improvement','room_for_improvement','room_for_improvement_pct'], var_name='metric', value_name='value')
plt.figure(figsize=(18,6),dpi=300)
display(df_metrics_melted)
sns.barplot(data=df_metrics_melted, x='fold', y='value', hue='metric')

# %%
# Target val f1 learning curves
curve_data = []
for experiment in tqdm(os.listdir(experiments_dir)):
    for run in os.listdir(f'{experiments_dir}/{experiment}'):
        if not os.path.exists(f'{experiments_dir}/{experiment}/{run}/losses.json'):
            continue
        losses = json.load(open(f'{experiments_dir}/{experiment}/{run}/losses.json'))
        hyperparameters = json.load(open(f'{experiments_dir}/{experiment}/{run}/hyperparameters.json'))

        if hyperparameters['mode'] != 'full_fine_tuning':
            continue
        
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

# %%



