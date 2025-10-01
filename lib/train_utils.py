import yaml
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import torch

def load_data(config):
    dataset_name = config['dataset'].get('name', 'default_dataset')
    data_path = f'data/{dataset_name}'
    target_participant = config['training'].get('target_participant', 'tj')
    batch_size = config['training'].get('batch_size', 512)
    fs = config['dataset']['fs']
    
    target_train_dataset = TensorDataset(*torch.load(f'{data_path}/{target_participant}_train.pt'))
    target_val_dataset = TensorDataset(*torch.load(f'{data_path}/{target_participant}_val.pt'))
    target_trainloader = DataLoader(target_train_dataset, batch_size=batch_size)
    target_valloader = DataLoader(target_val_dataset, batch_size=batch_size)

    print(f'Target train dataset size: {len(target_train_dataset)}')
    print(f'Target val dataset size: {len(target_val_dataset)}')
    # Print dataset length in hours using fs and window size and stride
    window_size_seconds = config['dataset'].get('window_size_seconds', 10)
    window_stride_seconds = config['dataset'].get('window_stride_seconds', 10)
    window_size_samples = window_size_seconds * fs
    window_stride_samples = window_stride_seconds * fs
    target_train_hours = (len(target_train_dataset) * window_stride_samples + (window_size_samples - window_stride_samples)) / fs / 3600
    target_val_hours = (len(target_val_dataset) * window_stride_samples + (window_size_samples - window_stride_samples)) / fs / 3600
    print(f'Target train dataset size: {target_train_hours:.2f} hours')
    print(f'Target val dataset size: {target_val_hours:.2f} hours')
    return target_trainloader, target_valloader

import json
def save_metrics_and_losses(metrics, lossi, config, new_exp_dir):
    with open(f'{new_exp_dir}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    with open(f'{new_exp_dir}/losses.json', 'w') as f:
        json.dump(lossi, f, indent=4)
    with open(f'{new_exp_dir}/hyperparameters.json', 'w') as f:
        json.dump(config, f, indent=4)

import matplotlib.pyplot as plt
def plot_loss_and_f1_refactored(lossi, new_exp_dir):
    fig,ax = plt.subplots(nrows=2,ncols=1,figsize=(7.2,9),dpi=300)
    if len(lossi['base train loss']) > 0 and len(lossi['base val loss']) > 0:
        ax[0].plot(lossi['base train loss'], label='Train (base)', color='b')
        ax[0].plot(lossi['base val loss'], label='Val (base)', color='b', linestyle='--')
    ax[0].plot(lossi['target train loss'], label='Train (target)', color='g')
    ax[0].plot(lossi['target val loss'], label='Val (target)', color='g', linestyle='--')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[0].set_yscale('log')

    if len(lossi['base train f1']) > 0 and len(lossi['base val f1']) > 0:
        ax[1].plot(lossi['base train f1'], color='b')
        ax[1].plot(lossi['base val f1'], color='b', linestyle='--')

    ax[1].plot(lossi['target train f1'], color='g')
    ax[1].plot(lossi['target val f1'], color='g', linestyle='--')

    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('F1')

    plt.savefig(f'{new_exp_dir}/loss_and_f1.jpg', bbox_inches='tight')
    plt.savefig(f'loss_and_f1.jpg', bbox_inches='tight')
    plt.close()
    
def random_subsample(dataset, pct):
    print(f'Subsampling dataset to {pct*100}% of original size')
    original_size = len(dataset)
    subset_size = int(original_size * pct)

    # Create random indices for subsampling
    import random
    random.seed(42)  # For reproducibility
    indices = random.sample(range(original_size), subset_size)

    # Create subset dataset
    from torch.utils.data import Subset
    return Subset(dataset, indices)

def append_losses_and_f1(phase, train_loss, train_f1, lossi, hyperparameters):
    if hyperparameters['mode'] == 'target_only':
        lossi['target train loss'].append(train_loss)
        lossi['target train f1'].append(train_f1)
    elif hyperparameters['mode'] == 'target_only_fine_tuning' and phase == 'base':
        lossi['target train loss'].append(None)
        lossi['target train f1'].append(None)

        lossi['base train loss'].append(train_loss)
        lossi['base train f1'].append(train_f1)
    elif hyperparameters['mode'] == 'target_only_fine_tuning' and phase == 'target':
        lossi['target train loss'].append(train_loss)
        lossi['target train f1'].append(train_f1)

        lossi['base train loss'].append(None)
        lossi['base train f1'].append(None)
    elif hyperparameters['mode'] == 'full_fine_tuning' and phase == 'base':
        lossi['base train loss'].append(train_loss)
        lossi['base train f1'].append(train_f1)

        lossi['target train loss'].append(None)
        lossi['target train f1'].append(None)
    elif hyperparameters['mode'] == 'full_fine_tuning' and phase == 'target':
        lossi['base train loss'].append(train_loss)
        lossi['base train f1'].append(train_f1)
    
    return lossi