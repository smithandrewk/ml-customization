import yaml
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import torch
from datetime import datetime

def smooth_curve(values, weight=0.9):
    """
    Apply exponential moving average (EMA) smoothing to a training curve.

    Args:
        values: List of numeric values (may contain None for unused epochs)
        weight: Smoothing factor (0-1). Higher = smoother curve.
                Default 0.9 provides ~10 epoch smoothing window.

    Returns:
        List of smoothed values (same length as input, None values preserved)
    """
    if not values or len(values) == 0:
        return values

    smoothed = []
    last_valid = None

    for val in values:
        if val is None:
            smoothed.append(None)
        else:
            if last_valid is None:
                # First valid value - no smoothing
                smoothed_val = val
            else:
                # EMA: smoothed = weight * prev_smoothed + (1 - weight) * current
                smoothed_val = weight * last_valid + (1 - weight) * val
            smoothed.append(smoothed_val)
            last_valid = smoothed_val

    return smoothed

def plot_curve_with_smoothing(ax, values, label, color, linestyle='-', weight=0.9, alpha_raw=0.3):
    """
    Plot both raw (transparent) and smoothed (prominent) versions of a curve.

    Args:
        ax: Matplotlib axis to plot on (if None, uses current axis via plt.gca())
        values: List of values to plot
        label: Label for the legend (applied to smoothed curve only)
        color: Color for both curves
        linestyle: Line style ('-', '--', etc.)
        weight: EMA smoothing weight
        alpha_raw: Transparency for raw data curve
    """
    if not values or len(values) == 0:
        return

    # Use current axis if not provided
    if ax is None:
        ax = plt.gca()

    # Plot raw data with transparency
    ax.plot(values, color=color, linestyle=linestyle, alpha=alpha_raw, linewidth=1)

    # Plot smoothed data prominently
    smoothed = smooth_curve(values, weight=weight)
    ax.plot(smoothed, label=label, color=color, linestyle=linestyle, alpha=1.0, linewidth=2)

def add_arguments(argparser):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    argparser.add_argument('--fold', type=int, required=False, default=0, help='Fold index for leave-one-participant-out cross-validation')
    argparser.add_argument('--device', type=int, required=False, default=0, help='GPU device index')
    argparser.add_argument('--batch_size', type=int, required=False, default=64, help='batch size')
    argparser.add_argument('--model', type=str, default='test', choices=['test'],help='Model architecture')
    argparser.add_argument('--use_augmentation', action='store_true', help='Enable data augmentation')
    argparser.add_argument('--jitter_std', type=float, default=0.005, help='Standard deviation for jitter noise')
    argparser.add_argument('--magnitude_range', type=float, nargs=2, default=[0.98, 1.02], help='Range for magnitude scaling')
    argparser.add_argument('--aug_prob', type=float, default=0.3, help='Probability of applying augmentation')
    argparser.add_argument('--dropout', type=float, default=0.5, help='Dropout probability (0.0 = no dropout)')
    argparser.add_argument('--use_dilation', action='store_true', help='Enable exponential dilation (1,2,4,8,...) in convolutional blocks')
    argparser.add_argument('--base_channels', type=int, default=8, help='Base number of channels in convolutional layers (8, 16, 32, etc.)')
    argparser.add_argument('--num_blocks', type=int, default=4, help='Number of convolutional blocks (depth of network)')
    argparser.add_argument('--use_residual', action='store_true', help='Enable residual connections (skip connections)')
    argparser.add_argument('--prefix', type=str, default=timestamp, help='Experiment prefix/directory name')
    argparser.add_argument('--early_stopping_patience', type=int, default=40, help='Early stopping patience for base phase')
    argparser.add_argument('--early_stopping_patience_target', type=int, default=40, help='Early stopping patience for target phase')
    argparser.add_argument('--early_stopping_metric', type=str, default='loss', choices=['loss', 'f1'], help='Metric to use for early stopping (loss or f1)')
    argparser.add_argument('--mode', type=str, default='full_fine_tuning', choices=['full_fine_tuning', 'target_only', 'target_only_fine_tuning'], help='Mode')
    argparser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    argparser.add_argument('--target_data_pct', type=float, default=1.0, help='Percentage of target training data to use (0.0-1.0)')
    argparser.add_argument('--participants', type=str, nargs='+', default=['tonmoy','asfik','ejaz'], help='List of participant names for cross-validation')
    argparser.add_argument('--window_size', type=int, default=3000, help='Window size in samples (e.g., 3000 = 60s at 50Hz)')
    argparser.add_argument('--data_path', type=str, default='data/001_60s_window', help='Path to dataset directory')
    argparser.add_argument('--n_base_participants', type=str, default='all', help='Number of base participants to use (integer or "all")')
    argparser.add_argument('--seed', type=int, default=42, help='Random seed for base model training')
    argparser.add_argument('--seed_finetune', type=int, default=None, help='Random seed for fine-tuning (if not set, uses --seed)')
    argparser.add_argument('--pos_weight', type=float, default=1, help='Positive class weight for BCE loss')
    return argparser

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def plot_loss_and_f1(lossi, new_exp_dir, metrics, patience_counter):
    plt.figure(figsize=(7.2,4.48),dpi=300)
    # Plot curves with smoothing
    plot_curve_with_smoothing(None, lossi['base train loss'], 'Train Loss (base)', color='b')
    plot_curve_with_smoothing(None, lossi['base val loss'], 'Val Loss (base)', color='b', linestyle='--')
    plot_curve_with_smoothing(None, lossi['target train loss'], 'Train Loss (target)', color='g')
    plot_curve_with_smoothing(None, lossi['target val loss'], 'Val Loss (target)', color='g', linestyle='--')

    if metrics['transition_epoch'] is not None:
        plt.axvline(x=metrics['transition_epoch'], color='r', linestyle='--', label='Phase Transition')

    if metrics['best_base_val_loss_epoch'] is not None and metrics['best_base_val_loss'] is not None:
        plt.axhline(y=metrics['best_base_val_loss'], color='b', linestyle='--', label='Best Base Val Loss', alpha=0.5)
        plt.axvline(x=metrics['best_base_val_loss_epoch'], color='b', linestyle='--', alpha=0.5)
        if metrics['best_base_val_loss_epoch'] < len(lossi['target val loss']) and lossi['target val loss'][metrics['best_base_val_loss_epoch']] is not None:
            plt.axhline(y=lossi['target val loss'][metrics['best_base_val_loss_epoch']], color='g', linestyle='--', label='Best Base Val Loss', alpha=0.5)

    if metrics['best_target_val_loss_epoch'] is not None and metrics['best_target_val_loss'] is not None:
        plt.axhline(y=metrics['best_target_val_loss'], color='g', linestyle='--', label='Best target Val Loss', alpha=0.8)
        plt.axvline(x=metrics['best_target_val_loss_epoch'], color='g', linestyle='--', alpha=0.4)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.yscale('log')
    plt.title(f'Patience Counter: {patience_counter}')
    plt.savefig(f'{new_exp_dir}/loss.jpg', bbox_inches='tight')
    plt.savefig(f'loss.jpg', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(7.2,4.48),dpi=300)
    # Plot curves with smoothing
    plot_curve_with_smoothing(None, lossi['base train f1'], 'Train f1 (base)', color='b')
    plot_curve_with_smoothing(None, lossi['base val f1'], 'Val f1 (base)', color='b', linestyle='--')
    plot_curve_with_smoothing(None, lossi['target train f1'], 'Train f1 (target)', color='g')
    plot_curve_with_smoothing(None, lossi['target val f1'], 'Val f1 (target)', color='g', linestyle='--')

    if metrics['transition_epoch'] is not None:
        plt.axvline(x=metrics['transition_epoch'], color='r', linestyle='--', label='Phase Transition')

    if metrics['best_base_val_f1_epoch'] is not None and metrics['best_base_val_f1'] is not None:
        plt.axhline(y=metrics['best_base_val_f1'], color='b', linestyle='--', label='Best Base f1 Loss', alpha=0.5)
        plt.axvline(x=metrics['best_base_val_f1_epoch'], color='b', linestyle='--', alpha=0.5)
        if metrics['best_base_val_f1_epoch'] < len(lossi['target val f1']) and lossi['target val f1'][metrics['best_base_val_f1_epoch']] is not None:
            plt.axhline(y=lossi['target val f1'][metrics['best_base_val_f1_epoch']], color='g', linestyle='--', label='Best Base Val Loss', alpha=0.5)

    if metrics['best_target_val_f1_epoch'] is not None and metrics['best_target_val_f1'] is not None:
        plt.axhline(y=metrics['best_target_val_f1'], color='g', linestyle='--', label='Best target f1 Loss', alpha=0.8)
        plt.axvline(x=metrics['best_target_val_f1_epoch'], color='g', linestyle='--', alpha=0.4)

    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.legend()
    plt.title(f'Patience Counter: {patience_counter}')
    plt.savefig(f'{new_exp_dir}/f1.jpg', bbox_inches='tight')
    plt.savefig(f'f1.jpg', bbox_inches='tight')
    plt.close()

def plot_base_training(lossi, new_exp_dir, metrics, patience_counter):
    """Plot training curves for base model training (no target phase)."""
    plt.figure(figsize=(7.2,4.48),dpi=300)
    # Plot curves with smoothing
    plot_curve_with_smoothing(None, lossi['train_loss'], 'Train Loss', color='b')
    plot_curve_with_smoothing(None, lossi['val_loss'], 'Val Loss', color='b', linestyle='--')

    if metrics['best_val_loss_epoch'] is not None and metrics['best_val_loss'] is not None:
        plt.axhline(y=metrics['best_val_loss'], color='b', linestyle='--', label='Best Val Loss', alpha=0.5)
        plt.axvline(x=metrics['best_val_loss_epoch'], color='b', linestyle='--', alpha=0.5)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.yscale('log')
    plt.title(f'Base Model Training - Patience: {patience_counter}')
    plt.savefig(f'{new_exp_dir}/loss.jpg', bbox_inches='tight')
    plt.savefig(f'loss.jpg', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(7.2,4.48),dpi=300)
    # Plot curves with smoothing
    plot_curve_with_smoothing(None, lossi['train_f1'], 'Train F1', color='b')
    plot_curve_with_smoothing(None, lossi['val_f1'], 'Val F1', color='b', linestyle='--')

    if metrics['best_val_f1_epoch'] is not None and metrics['best_val_f1'] is not None:
        plt.axhline(y=metrics['best_val_f1'], color='b', linestyle='--', label='Best Val F1', alpha=0.5)
        plt.axvline(x=metrics['best_val_f1_epoch'], color='b', linestyle='--', alpha=0.5)

    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.legend()
    plt.title(f'Base Model Training - Patience: {patience_counter}')
    plt.savefig(f'{new_exp_dir}/f1.jpg', bbox_inches='tight')
    plt.savefig(f'f1.jpg', bbox_inches='tight')
    plt.close()

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

def plot_loss_and_f1_refactored(lossi, new_exp_dir):
    fig,ax = plt.subplots(nrows=2,ncols=1,figsize=(7.2,9),dpi=300)

    # Plot loss curves with smoothing
    if len(lossi['base train loss']) > 0 and len(lossi['base val loss']) > 0:
        plot_curve_with_smoothing(ax[0], lossi['base train loss'], 'Train (base)', color='b')
        plot_curve_with_smoothing(ax[0], lossi['base val loss'], 'Val (base)', color='b', linestyle='--')
    plot_curve_with_smoothing(ax[0], lossi['target train loss'], 'Train (target)', color='g')
    plot_curve_with_smoothing(ax[0], lossi['target val loss'], 'Val (target)', color='g', linestyle='--')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[0].set_yscale('log')

    # Plot F1 curves with smoothing
    if len(lossi['base train f1']) > 0 and len(lossi['base val f1']) > 0:
        plot_curve_with_smoothing(ax[1], lossi['base train f1'], 'Train (base)', color='b')
        plot_curve_with_smoothing(ax[1], lossi['base val f1'], 'Val (base)', color='b', linestyle='--')

    plot_curve_with_smoothing(ax[1], lossi['target train f1'], 'Train (target)', color='g')
    plot_curve_with_smoothing(ax[1], lossi['target val f1'], 'Val (target)', color='g', linestyle='--')

    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('F1')
    ax[1].legend()

    plt.savefig(f'{new_exp_dir}/loss_and_f1.jpg', bbox_inches='tight')
    plt.savefig(f'loss_and_f1.jpg', bbox_inches='tight')
    plt.close()
    
def random_subsample(dataset, pct):
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


from sklearn.metrics import f1_score

def compute_loss_and_f1(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    count = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for Xi, yi in dataloader:
            Xi = Xi.to(device)
            yi = yi.to(device).float()
            logits = model(Xi).squeeze(-1)
            loss = criterion(logits, yi)
            total_loss += loss.item() * Xi.size(0)
            count += Xi.size(0)
            y_true.append(yi.cpu())
            y_pred.append(logits.sigmoid().round().cpu())
    y_true = torch.cat(y_true).cpu()
    y_pred = torch.cat(y_pred).cpu()
    f1 = f1_score(y_true, y_pred, average='macro')

    return total_loss / count, float(f1)

def compute_loss_and_f1_and_precision_and_recall(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    count = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for Xi, yi in dataloader:
            Xi = Xi.to(device)
            yi = yi.to(device).float()
            logits = model(Xi).squeeze(-1)
            loss = criterion(logits, yi)
            total_loss += loss.item() * Xi.size(0)
            count += Xi.size(0)
            y_true.append(yi.cpu())
            y_pred.append(logits.sigmoid().round().cpu())
    y_true = torch.cat(y_true).cpu()
    y_pred = torch.cat(y_pred).cpu()
    f1 = f1_score(y_true, y_pred, average='macro')
    from sklearn.metrics import precision_score, recall_score
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)

    return total_loss / count, float(f1), float(precision), float(recall)

def optimize_model_compute_loss_and_f1(model, dataloader, optimizer, criterion, device, augmenter=None):
    model.train()
    total_loss = 0.0
    count = 0
    y_true = []
    y_pred = []
    for Xi, yi in dataloader:
        Xi = Xi.to(device)
        yi = yi.to(device).float().view(-1,1)

        # Apply augmentation if provided
        if augmenter is not None:
            Xi, yi = augmenter.augment(Xi, yi)

        optimizer.zero_grad()
        logits = model(Xi)
        loss = criterion(logits,yi)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * Xi.size(0)
        count += Xi.size(0)
        y_true.append(yi.cpu())
        y_pred.append(logits.sigmoid().round().cpu())
    y_true = torch.cat(y_true).cpu().detach()
    y_pred = torch.cat(y_pred).cpu().detach()
    f1 = f1_score(y_true, y_pred, average='macro')
    # f1 = (2 * (y_true * y_pred).sum()) / ((y_true + y_pred).sum() + 1e-8)
    return total_loss / count, float(f1)

from sklearn.metrics import classification_report,ConfusionMatrixDisplay

def evaluate(model, dataloader, device):
    y_pred = []
    y_true = []
    model.eval()
    model.to(device)
    with torch.no_grad():
        for Xi,yi in dataloader:
            Xi = Xi.to(device)
            y_true.append(yi)
            y_pred.append(model(Xi).sigmoid().round().cpu().flatten())
    y_true = torch.cat(y_true).cpu()
    y_pred = torch.cat(y_pred).cpu()

    print(classification_report(y_true, y_pred, target_names=['No Smoking', 'Smoking']))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred,normalize='true')
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred,normalize='pred')
    return y_true,y_pred

class TimeSeriesAugmenter:
    """Time-series specific data augmentation for accelerometer data."""
    
    def __init__(self, config):
        self.jitter_std = config.get('jitter_noise_std', 0.01)
        self.magnitude_range = config.get('magnitude_scale_range', [0.9, 1.1])
        self.time_warp_sigma = config.get('time_warp_sigma', 0.2)
        self.prob = config.get('augmentation_probability', 0.5)
    
    def jitter(self, X):
        """Add random noise to time series."""
        noise = torch.randn_like(X) * self.jitter_std
        return X + noise
    
    def magnitude_scale(self, X):
        """Scale magnitude of time series."""
        scale = torch.FloatTensor(1).uniform_(*self.magnitude_range).item()
        return X * scale
    
    def time_warp(self, X):
        """Apply time warping to time series."""
        batch_size, seq_len, features = X.shape
        
        # Create random warp factors
        warp_steps = max(1, int(seq_len * 0.1))  # 10% of sequence length
        warp_locs = torch.randint(warp_steps, seq_len - warp_steps, (batch_size,))
        warp_factors = torch.normal(1.0, self.time_warp_sigma, (batch_size,))
        
        # Apply warping (simplified version)
        warped_X = X.clone()
        for i in range(batch_size):
            if torch.rand(1) < 0.5:  # 50% chance to apply warping
                loc = warp_locs[i]
                factor = warp_factors[i]
                
                # Simple implementation: just scale a portion of the signal
                start_idx = max(0, loc - warp_steps // 2)
                end_idx = min(seq_len, loc + warp_steps // 2)
                warped_X[i, start_idx:end_idx] *= factor
        
        return warped_X
    
    def augment(self, X, y):
        """Apply random augmentation to batch."""
        if torch.rand(1) > self.prob:
            return X, y
        
        X_aug = X.clone()
        
        # Randomly select augmentation type
        aug_type = torch.randint(0, 3, (1,)).item()
        
        if aug_type == 0:
            X_aug = self.jitter(X_aug)
        elif aug_type == 1:
            X_aug = self.magnitude_scale(X_aug)
        else:  # aug_type == 2
            X_aug = self.time_warp(X_aug)
        
        return X_aug, y

import seaborn as sns
def plot_hyperparameter_counts(df, hyperparameters_to_plot, max_combinations=10):
    """
    Create heatmaps showing counts of hyperparameter combinations
    """
    if len(hyperparameters_to_plot) < 2:
        print("Need at least 2 hyperparameters for combination plots")
        return
    
    # Generate all pairs of hyperparameters
    from itertools import combinations
    param_pairs = list(combinations(hyperparameters_to_plot, 2))
    
    if len(param_pairs) > max_combinations:
        print(f"Too many combinations ({len(param_pairs)}), showing first {max_combinations}")
        param_pairs = param_pairs[:max_combinations]
    
    # Calculate grid dimensions
    n_pairs = len(param_pairs)
    n_cols = min(3, n_pairs)
    n_rows = (n_pairs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    
    # Handle single subplot case
    if n_pairs == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    axes_flat = axes.flatten() if n_pairs > 1 else axes
    
    for i, (param1, param2) in enumerate(param_pairs):
        ax = axes_flat[i]
        
        # Check if parameters exist in dataframe
        if param1 not in df.columns or param2 not in df.columns:
            ax.text(0.5, 0.5, f'Parameters "{param1}" or "{param2}"\nnot found in data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{param1} vs {param2} (Not Found)')
            continue
        
        # Create combination counts
        combo_df = df.groupby([param1, param2]).size().reset_index(name='count')
        
        # Create pivot table
        pivot_df = combo_df.pivot_table(
            values='count', 
            index=param1, 
            columns=param2, 
            fill_value=0
        )
        
        # Create heatmap
        sns.heatmap(pivot_df, annot=pivot_df.astype(str), cmap='Blues', ax=ax, fmt='s')
        ax.set_title(f'{param1} vs {param2} Counts')
    
    # Hide unused subplots
    for i in range(n_pairs, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

from tqdm import tqdm
import os
from torch import nn
import pandas as pd

def load_experiments_from_dir(experiments_dir, device='cuda'):
    data = []

    for experiment in tqdm(os.listdir(experiments_dir)):
        for run in os.listdir(f'{experiments_dir}/{experiment}'):
            if not os.path.exists(f'{experiments_dir}/{experiment}/{run}/metrics.json'):
                print(f'Skipping {experiments_dir}/{experiment}/{run} as no metrics.json')
                continue
            
            metrics = json.load(open(f'{experiments_dir}/{experiment}/{run}/metrics.json'))
            losses = json.load(open(f'{experiments_dir}/{experiment}/{run}/losses.json'))
            hyperparameters = json.load(open(f'{experiments_dir}/{experiment}/{run}/hyperparameters.json'))


            if experiment.startswith('base'):
                if 'test_f1' in metrics and 'test_loss' in metrics and 'test_precision' in metrics and 'test_recall' in metrics and 'target_val_f1' in metrics and 'target_val_loss' in metrics and 'target_val_precision' in metrics and 'target_val_recall' in metrics:
                    # Already computed test metrics
                    pass
                else:
                    hyperparameters['mode'] = 'base'
                    best_base_model_path = f'{experiments_dir}/{experiment}/{run}/best_base_model.pt'
                    from lib.models import TestModel
                    target_participant = hyperparameters['target_participant']
                    data_path = hyperparameters['data_path']
                    batch_size = hyperparameters['batch_size']
                    criterion = nn.BCEWithLogitsLoss()

                    target_test_dataset = TensorDataset(*torch.load(f'{data_path}/{target_participant}_test.pt'))
                    target_testloader = DataLoader(target_test_dataset, batch_size=batch_size, shuffle=False)
                    model = TestModel(dropout=hyperparameters['dropout'],
                                    use_dilation=hyperparameters['use_dilation'],
                                    base_channels=hyperparameters['base_channels'],
                                    num_blocks=hyperparameters['num_blocks'],
                                    use_residual=hyperparameters['use_residual'])
                    model.load_state_dict(torch.load(best_base_model_path, map_location='cpu'))
                    model.to(device)
                    test_loss, test_f1, test_precision, test_recall = compute_loss_and_f1_and_precision_and_recall(model, target_testloader, criterion, device=device)
                    metrics['test_f1'] = test_f1
                    metrics['test_loss'] = test_loss
                    metrics['test_precision'] = test_precision
                    metrics['test_recall'] = test_recall

                    target_val_dataset = TensorDataset(*torch.load(f'{data_path}/{target_participant}_val.pt'))
                    target_valloader = DataLoader(target_val_dataset, batch_size=batch_size, shuffle=False)
                    model = TestModel(dropout=hyperparameters['dropout'],
                                    use_dilation=hyperparameters['use_dilation'],
                                    base_channels=hyperparameters['base_channels'],
                                    num_blocks=hyperparameters['num_blocks'],
                                    use_residual=hyperparameters['use_residual'])
                    model.load_state_dict(torch.load(best_base_model_path, map_location='cpu'))
                    model.to(device)
                    val_loss, val_f1, val_precision, val_recall = compute_loss_and_f1_and_precision_and_recall(model, target_valloader, criterion, device=device)
                    metrics['target_val_f1'] = val_f1
                    metrics['target_val_loss'] = val_loss
                    metrics['target_val_precision'] = val_precision
                    metrics['target_val_recall'] = val_recall

                    with open(f'{experiments_dir}/{experiment}/{run}/metrics.json', 'w') as f:
                        json.dump(metrics, f, indent=4)
                    with open(f'{experiments_dir}/{experiment}/{run}/hyperparameters.json', 'w') as f:
                        json.dump(hyperparameters, f, indent=4)
            elif experiment.startswith('finetune') or experiment.startswith('target_only'):
                if 'test_f1' in metrics and 'test_loss' in metrics and 'test_precision' in metrics and 'test_recall' in metrics:
                    # Already computed test metrics
                    pass
                else:
                    best_model_path = f'{experiments_dir}/{experiment}/{run}/best_model.pt'
                    from lib.models import TestModel
                    target_participant = hyperparameters['target_participant']
                    data_path = hyperparameters['data_path']
                    batch_size = hyperparameters['batch_size']
                    criterion = nn.BCEWithLogitsLoss()

                    target_test_dataset = TensorDataset(*torch.load(f'{data_path}/{target_participant}_test.pt'))
                    target_testloader = DataLoader(target_test_dataset, batch_size=batch_size, shuffle=False)
                    model = TestModel(dropout=hyperparameters['dropout'],
                                    use_dilation=hyperparameters['use_dilation'],
                                    base_channels=hyperparameters['base_channels'],
                                    num_blocks=hyperparameters['num_blocks'],
                                    use_residual=hyperparameters['use_residual'])
                    model.load_state_dict(torch.load(best_model_path, map_location='cpu'))
                    model.to(device)
                    test_loss, test_f1, test_precision, test_recall = compute_loss_and_f1_and_precision_and_recall(model, target_testloader, criterion, device=device)
                    metrics['test_f1'] = test_f1
                    metrics['test_loss'] = test_loss
                    metrics['test_precision'] = test_precision
                    metrics['test_recall'] = test_recall
                    with open(f'{experiments_dir}/{experiment}/{run}/metrics.json', 'w') as f:
                        json.dump(metrics, f, indent=4)
                
            data.append({
                **metrics,
                **hyperparameters,
                'experiment': experiment,
                'run': run
            })
            
    df = pd.DataFrame(data)
    return df


