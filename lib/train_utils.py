import yaml
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import torch
from datetime import datetime
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
    argparser.add_argument('--prefix', type=str, default=timestamp, help='Experiment prefix/directory name')
    argparser.add_argument('--early_stopping_patience', type=int, default=40, help='Early stopping patience for base phase')
    argparser.add_argument('--early_stopping_patience_target', type=int, default=40, help='Early stopping patience for target phase')
    argparser.add_argument('--mode', type=str, default='full_fine_tuning', choices=['full_fine_tuning', 'target_only', 'target_only_fine_tuning'], help='Mode')
    argparser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    argparser.add_argument('--target_data_pct', type=float, default=1.0, help='Percentage of target training data to use (0.0-1.0)')
    argparser.add_argument('--participants', type=str, nargs='+', default=['tonmoy','asfik','ejaz'], help='List of participant names for cross-validation')
    argparser.add_argument('--window_size', type=int, default=3000, help='Window size in samples (e.g., 3000 = 60s at 50Hz)')
    argparser.add_argument('--data_path', type=str, default='data/001_60s_window', help='Path to dataset directory')
    argparser.add_argument('--n_base_participants', type=str, default='all', help='Number of base participants to use (integer or "all")')
    argparser.add_argument('--seed', type=int, default=42, help='Random seed for base model training')
    argparser.add_argument('--seed_finetune', type=int, default=None, help='Random seed for fine-tuning (if not set, uses --seed)')
    return argparser

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def plot_loss_and_f1(lossi, new_exp_dir, metrics, patience_counter):
    plt.figure(figsize=(7.2,4.48),dpi=300)
    plt.plot(lossi['base train loss'], label='Train Loss (base)', color='b')
    plt.plot(lossi['base val loss'], label='Val Loss (base)', color='b', linestyle='--')
    plt.plot(lossi['target train loss'], label='Train Loss (target)', color='g')
    plt.plot(lossi['target val loss'], label='Val Loss (target)', color='g', linestyle='--')

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
    plt.plot(lossi['base train f1'], label='Train f1 (base)', color='b')
    plt.plot(lossi['base val f1'], label='Val f1 (base)', color='b', linestyle='--')
    plt.plot(lossi['target train f1'], label='Train f1 (target)', color='g')
    plt.plot(lossi['target val f1'], label='Val f1 (target)', color='g', linestyle='--')

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
    plt.plot(lossi['train_loss'], label='Train Loss', color='b')
    plt.plot(lossi['val_loss'], label='Val Loss', color='b', linestyle='--')

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
    plt.plot(lossi['train_f1'], label='Train F1', color='b')
    plt.plot(lossi['val_f1'], label='Val F1', color='b', linestyle='--')

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