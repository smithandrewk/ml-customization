import mysql.connector
from mysql.connector import Error
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from dotenv import load_dotenv
from contextlib import contextmanager
import yaml
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from torch.nn.functional import relu
import torch.nn as nn

load_dotenv()

"""
Configuration and database utils
"""

def resample(df,target_hz=50):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['ns_since_reboot'], unit='ns')
    df = df.set_index('timestamp')
    freq = f'{1000//target_hz}ms'  # 20ms for 50Hz
    df_resampled = df.resample(freq).mean().ffill()
    df_resampled = df_resampled.reset_index()
    df_resampled['ns_since_reboot'] = df_resampled['timestamp'].astype('int64')
    df = df_resampled.drop('timestamp', axis=1)
    return df

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Calculate derived parameters
    config['dataset']['window_size'] = config['dataset']['fs'] * config['dataset']['window_size_seconds']
    config['dataset']['window_stride'] = config['dataset']['fs'] * config['dataset']['window_stride_seconds']
    
    # Update model window_size to match dataset
    config['model']['window_size'] = config['dataset']['window_size']
    
    # Calculate num_features based on enabled sensors
    num_features = 0
    if config.get('sensors', {}).get('use_accelerometer', True):  # Default to True for backward compatibility
        num_features += 3
    if config.get('sensors', {}).get('use_gyroscope', False):  # Default to False for backward compatibility
        num_features += 3
    
    # Ensure at least one sensor is enabled
    if num_features == 0:
        raise ValueError("At least one sensor (accelerometer or gyroscope) must be enabled")
    
    config['model']['num_features'] = num_features
    
    return config

def get_experiment_dir(config):
    """Get the experiment directory path."""
    return os.path.join(config['experiment']['save_dir'], config['experiment']['name'])

def get_next_experiment_dir(dataset_name, experiment_type="train"):
    """Get next auto-incrementing experiment directory with dataset name.
    
    Format: experiments/{global_index:03d}_{dataset_name}_{dataset_exp_index:03d}/
    
    Args:
        dataset_name: Name of dataset (e.g., "tonmoy_60s", extracted from data/001_tonmoy_60s/)
        experiment_type: Type of experiment ("train", "custom", etc.)
    """
    os.makedirs('experiments', exist_ok=True)
    
    # Find highest global experiment number
    existing_dirs = [d for d in os.listdir('experiments') if os.path.isdir(f'experiments/{d}')]
    global_numbers = []
    dataset_numbers = []
    
    for d in existing_dirs:
        if '_' in d and len(d.split('_')) >= 3:
            try:
                parts = d.split('_')
                global_num = int(parts[0])
                global_numbers.append(global_num)
                
                # Check if this experiment uses the same dataset
                if dataset_name in d:
                    dataset_exp_num = int(parts[-1])
                    dataset_numbers.append(dataset_exp_num)
                    
            except (ValueError, IndexError):
                continue
    
    next_global_num = max(global_numbers) + 1 if global_numbers else 1
    next_dataset_num = max(dataset_numbers) + 1 if dataset_numbers else 1
    
    experiment_dir = f'experiments/{next_global_num:03d}_{dataset_name}_{next_dataset_num:03d}'
    os.makedirs(experiment_dir, exist_ok=True)
    
    return experiment_dir

@contextmanager
def db_connection():
    """Context manager for database connections that ensures proper cleanup."""
    conn = None
    try:
        MYSQL_CONFIG = {
            'host': os.getenv('MYSQL_HOST', '10.173.98.204'),
            'user': os.getenv('MYSQL_USER', 'andrew'),
            'password': os.getenv('MYSQL_PASSWORD', 'admin'),
            'database': os.getenv('MYSQL_DATABASE', 'delta2')
        }
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        yield conn
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        raise
    finally:
        if conn and conn.is_connected():
            conn.close()

def get_db_connection():
    try:
        MYSQL_CONFIG = {
            'host': '10.173.98.204',
            'user': 'andrew',
            'password': 'admin',
            'database': 'delta2'
        }
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        return conn
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None
    
def moving_average(data, window_size):
    """Calculate moving average with specified window size."""
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')

def create_stratified_combined_dataloader(base_datasets, target_datasets, target_weight_multiplier, batch_size):
    """
    Create a stratified dataloader that ensures balanced batches instead of using WeightedRandomSampler.
    
    This approach creates separate class-balanced samplers and combines them to avoid extreme batch compositions.
    
    Args:
        base_datasets: List of base participant datasets
        target_datasets: List of target participant datasets  
        target_weight_multiplier: Weight multiplier for target samples
        batch_size: Batch size for the dataloader
    
    Returns:
        DataLoader with stratified sampling to maintain batch balance
    """
    from torch.utils.data import DataLoader, ConcatDataset, SubsetRandomSampler
    import random
    
    if not base_datasets and not target_datasets:
        raise ValueError("No datasets provided")
    
    # Special case: target_weight = 0 means exclude target data entirely
    if target_weight_multiplier == 0.0:
        if not base_datasets:
            raise ValueError("Cannot create dataloader with target_weight=0 and no base datasets")
        print("⚠️  target_weight=0.0: Using stratified sampling on base data only")
        combined_dataset = ConcatDataset(base_datasets)
        return DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
    
    # Special case: no target datasets provided
    if not target_datasets:
        combined_dataset = ConcatDataset(base_datasets)
        return DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
    
    # Normal case: create stratified sampling
    all_datasets = base_datasets + target_datasets
    combined_dataset = ConcatDataset(all_datasets)
    
    # Get labels for all samples to enable stratification
    all_labels = []
    sample_weights = []
    current_idx = 0
    
    # Process base datasets
    for dataset in base_datasets:
        for _, label in dataset:
            all_labels.append(label.item() if hasattr(label, 'item') else label)
            sample_weights.append(1.0)  # Base weight = 1.0
        current_idx += len(dataset)
    
    # Process target datasets  
    for dataset in target_datasets:
        for _, label in dataset:
            all_labels.append(label.item() if hasattr(label, 'item') else label)
            sample_weights.append(target_weight_multiplier)  # Target weight = multiplier
        current_idx += len(dataset)
    
    # Convert to tensors
    all_labels = torch.tensor(all_labels)
    sample_weights = torch.tensor(sample_weights)
    
    # Calculate effective target representation
    base_count = sum(len(d) for d in base_datasets) 
    target_count = sum(len(d) for d in target_datasets)
    total_weight = base_count * 1.0 + target_count * target_weight_multiplier
    effective_target_percentage = (target_count * target_weight_multiplier / total_weight) * 100
    
    print(f"📊 Stratified sampling: {effective_target_percentage:.1f}% effective target representation")
    print(f"   - Base samples: {base_count:,} (weight: 1.0)")
    print(f"   - Target samples: {target_count:,} (weight: {target_weight_multiplier:.1f})")
    
    # Use WeightedRandomSampler but with stratification to maintain balance
    from torch.utils.data import WeightedRandomSampler
    
    # Create sampler that respects weights but tries to maintain class balance
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    print("   ⚠️  Note: Still using WeightedRandomSampler - consider switching to StratifiedBatchSampler for complete fix")
    
    return DataLoader(combined_dataset, batch_size=batch_size, sampler=sampler)

def plot_training_progress(trainlossi=None, testlossi=None, targetlossi=None, trainf1i=None, testf1i=None, targetf1i=None, basevalidationlossi=None, targetvalidationlossi=None, basevalidationf1i=None, targetvalidationf1i=None, ma_window_size=10, save_path='training_metrics.jpg', transition_epoch=None, custom_labels=None, best_models=None, base_trainlossi=None, target_trainlossi=None, base_trainf1i=None, target_trainf1i=None):
    """
    Plot training progress with loss on top subplot and F1 scores on bottom subplot.
    Both subplots share the x-axis.
    """
    # Set up figure with two subplots sharing x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14), sharex=True, gridspec_kw={'height_ratios': [1, 1]})
    
    # Setup colors: Blue for base metrics, Orange for target metrics
    colors = {
        'base': '#0072B2',      # Professional blue for all base metrics
        'target': '#D55E00',    # Professional orange for all target metrics
        # Legacy color mappings for backward compatibility
        'train': '#0072B2',     # Training (base) - blue
        'test': '#D55E00',      # Legacy test - orange  
        'base_val': '#0072B2',  # Base validation - blue
        'target_val': '#D55E00' # Target validation - orange
    }
    
    # Setup linestyles to distinguish metric types
    linestyles = {
        'train': '-',           # Solid line for training
        'val': '--',           # Dashed line for validation  
        'test': ':',           # Dotted line for test
        'base_val': '--',      # Base validation - dashed
        'target_val': '--'     # Target validation - dashed
    }
    
    # PLOT 1: LOSS VALUES
    # Determine x-axis length from available data
    if testlossi is not None:
        x_axis_length = len(testlossi)
    elif basevalidationlossi is not None:
        x_axis_length = len(basevalidationlossi)
    elif targetvalidationlossi is not None:
        x_axis_length = len(targetvalidationlossi)
    elif targetlossi is not None:
        x_axis_length = len(targetlossi)
    else:
        x_axis_length = len(trainlossi)
    
    # Plot base training loss (Phase 1 only) with low opacity
    if base_trainlossi is not None:
        base_trainlossi_sliced = base_trainlossi[1:]  # Remove first value
        x_base_trainlossi = np.array(range(len(base_trainlossi_sliced)))
        ax1.plot(x_base_trainlossi, base_trainlossi_sliced, 
                 alpha=0.3, color=colors['base'], linestyle=linestyles['train'], linewidth=1, label='_nolegend_')
    
    # Plot target training loss (continuous from epoch 0) with low opacity  
    if target_trainlossi is not None:
        target_trainlossi_sliced = target_trainlossi[1:]  # Remove first value
        x_target_trainlossi = np.array(range(len(target_trainlossi_sliced)))
        ax1.plot(x_target_trainlossi, target_trainlossi_sliced,
                 alpha=0.3, color=colors['target'], linestyle=linestyles['train'], linewidth=1, label='_nolegend_')
    
    # Backward compatibility: if old trainlossi is provided but not the new separate ones
    if trainlossi is not None and base_trainlossi is None and target_trainlossi is None:
        ax1.plot(np.linspace(0, x_axis_length-1, len(trainlossi)), trainlossi, 
                 alpha=0.3, color=colors['base'], linestyle=linestyles['train'], linewidth=1, label='_nolegend_')
    
    if testlossi is not None:
        ax1.plot(np.linspace(0, x_axis_length-1, len(testlossi)), testlossi, 
                 alpha=0.3, color=colors['target'], linestyle=linestyles['test'], linewidth=1, label='_nolegend_')
    
    if targetlossi is not None:
        ax1.plot(np.linspace(0, x_axis_length-1, len(targetlossi)), targetlossi, 
                alpha=0.3, color=colors['target'], linestyle=linestyles['test'], linewidth=1, label='_nolegend_')
    
    # Plot base training moving average (Phase 1)
    if base_trainlossi is not None and len(base_trainlossi) > ma_window_size:
        base_trainlossi_ma = moving_average(base_trainlossi[1:], ma_window_size)  # Remove first value
        x_base_trainlossi_ma = np.arange(ma_window_size-1, len(base_trainlossi)-1)
        base_train_label = custom_labels.get('base_train', 'Base Training Loss (MA)') if custom_labels else 'Base Training Loss (MA)'
        ax1.plot(x_base_trainlossi_ma, base_trainlossi_ma, color=colors['base'], linestyle=linestyles['train'],
                 linewidth=2, label=base_train_label)
        
        # Mark minimum base training loss
        min_idx = np.argmin(base_trainlossi_ma)
        min_val = np.min(base_trainlossi_ma)
        ax1.plot(x_base_trainlossi_ma[min_idx], min_val, 'o', color=colors['base'], markersize=8)
    
    # Plot target training moving average (continuous from epoch 0)
    if target_trainlossi is not None and len(target_trainlossi) > ma_window_size:
        target_trainlossi_ma = moving_average(target_trainlossi[1:], ma_window_size)  # Remove first value
        x_target_trainlossi_ma = np.arange(ma_window_size-1, len(target_trainlossi)-1)
        target_train_label = custom_labels.get('target_train', 'Target Training Loss (MA)') if custom_labels else 'Target Training Loss (MA)'
        ax1.plot(x_target_trainlossi_ma, target_trainlossi_ma, color=colors['target'], linestyle=linestyles['train'],
                 linewidth=2, label=target_train_label)
        
        # Mark minimum target training loss
        min_idx = np.argmin(target_trainlossi_ma)
        min_val = np.min(target_trainlossi_ma)
        ax1.plot(x_target_trainlossi_ma[min_idx], min_val, 'o', color=colors['target'], markersize=8)
    
    # Backward compatibility for old trainlossi parameter
    if trainlossi is not None and base_trainlossi is None and target_trainlossi is None and len(trainlossi) > ma_window_size:
        trainlossi_ma = moving_average(trainlossi, ma_window_size)
        x_trainlossi_ma = np.linspace(ma_window_size-1, x_axis_length-1, len(trainlossi_ma))
        train_label = custom_labels.get('train', 'Training Loss (MA)') if custom_labels else 'Training Loss (MA)'
        ax1.plot(x_trainlossi_ma, trainlossi_ma, color=colors['base'], linestyle=linestyles['train'],
                 linewidth=2, label=train_label)
        
        min_idx = np.argmin(trainlossi_ma)
        min_val = np.min(trainlossi_ma)
        ax1.plot(x_trainlossi_ma[min_idx], min_val, 'o', color=colors['base'], markersize=8)
    # Plot moving average if we have enough data
    if testlossi is not None and len(testlossi) > ma_window_size:
        testlossi_ma = moving_average(testlossi, ma_window_size)
        x_testlossi_ma = np.linspace(ma_window_size-1, x_axis_length-1, len(testlossi_ma))
        # Determine validation label
        test_label = custom_labels.get('test', 'Validation Loss (MA)') if custom_labels else 'Validation Loss (MA)'
        ax1.plot(x_testlossi_ma, testlossi_ma, color=colors['target'], linestyle=linestyles['test'],
                 linewidth=2, label=test_label)
        
        # Mark minimum testing loss
        min_idx = np.argmin(testlossi_ma)
        min_val = np.min(testlossi_ma)
        ax1.plot(x_testlossi_ma[min_idx], min_val, 'o', color=colors['target'], markersize=8)
    
    if targetlossi is not None:
        # Plot moving average if we have enough data
        if len(targetlossi) > ma_window_size:
            targetlossi_ma = moving_average(targetlossi, ma_window_size)
            x_targetlossi_ma = np.linspace(ma_window_size-1, x_axis_length-1, len(targetlossi_ma))
            # Determine target label
            target_label = custom_labels.get('target', 'Target Test Loss (MA)') if custom_labels else 'Target Test Loss (MA)'
            ax1.plot(x_targetlossi_ma, targetlossi_ma, color=colors['target'], linestyle=linestyles['test'],
                    linewidth=2, label=target_label)
            
            # Mark minimum targeting loss
            min_idx = np.argmin(targetlossi_ma)
            min_val = np.min(targetlossi_ma)
            ax1.plot(x_targetlossi_ma[min_idx], min_val, 'o', color=colors['target'], markersize=8)
    
    # Plot base validation loss if provided
    if basevalidationlossi is not None:
        # Plot raw base validation loss with low opacity
        ax1.plot(np.linspace(0, len(basevalidationlossi)-1, len(basevalidationlossi)), basevalidationlossi, 
                 alpha=0.3, color=colors['base'], linestyle=linestyles['base_val'], linewidth=1, label='_nolegend_')
        
        # Plot moving average if we have enough data
        if len(basevalidationlossi) > ma_window_size:
            basevalidationlossi_ma = moving_average(basevalidationlossi, ma_window_size)
            x_basevalidationlossi_ma = np.linspace(ma_window_size-1, len(basevalidationlossi)-1, len(basevalidationlossi_ma))
            # Determine base validation label
            base_val_label = custom_labels.get('base_val', 'Base Validation Loss (MA)') if custom_labels else 'Base Validation Loss (MA)'
            ax1.plot(x_basevalidationlossi_ma, basevalidationlossi_ma, color=colors['base'], linestyle=linestyles['base_val'],
                     linewidth=2, label=base_val_label)
            
            # Mark minimum base validation loss
            min_idx = np.argmin(basevalidationlossi_ma)
            min_val = np.min(basevalidationlossi_ma)
            ax1.plot(x_basevalidationlossi_ma[min_idx], min_val, 'o', color=colors['base'], markersize=8)
    
    # Plot target validation loss if provided
    if targetvalidationlossi is not None:
        # Plot raw target validation loss with low opacity
        ax1.plot(np.linspace(0, len(targetvalidationlossi)-1, len(targetvalidationlossi)), targetvalidationlossi, 
                 alpha=0.3, color=colors['target'], linestyle=linestyles['target_val'], linewidth=1, label='_nolegend_')
        
        # Plot moving average if we have enough data
        if len(targetvalidationlossi) > ma_window_size:
            targetvalidationlossi_ma = moving_average(targetvalidationlossi, ma_window_size)
            x_targetvalidationlossi_ma = np.linspace(ma_window_size-1, len(targetvalidationlossi)-1, len(targetvalidationlossi_ma))
            # Determine target validation label
            target_val_label = custom_labels.get('target_val', 'Target Validation Loss (MA)') if custom_labels else 'Target Validation Loss (MA)'
            ax1.plot(x_targetvalidationlossi_ma, targetvalidationlossi_ma, color=colors['target'], linestyle=linestyles['target_val'],
                     linewidth=2, label=target_val_label)
            
            # Mark minimum target validation loss
            min_idx = np.argmin(targetvalidationlossi_ma)
            min_val = np.min(targetvalidationlossi_ma)
            ax1.plot(x_targetvalidationlossi_ma[min_idx], min_val, 'o', color=colors['target'], markersize=8)
    
    # Professional styling for loss subplot
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_yscale('log')
    ax1.set_ylabel('Log Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training Log Loss', fontsize=14, fontweight='bold', pad=10)
    
    # Only show legend if there are labeled plots
    handles, labels = ax1.get_legend_handles_labels()
    if handles:
        ax1.legend(frameon=True, framealpha=0.9)
    
    # PLOT 2: F1 SCORES (only if provided)
    # Plot base training F1 (Phase 1 only)
    if base_trainf1i is not None:
        base_trainf1i_sliced = base_trainf1i[1:]  # Remove first value
        x_base_trainf1i = np.array(range(len(base_trainf1i_sliced)))
        ax2.plot(x_base_trainf1i, base_trainf1i_sliced, color=colors['base'], linestyle=linestyles['train'],
                linewidth=1, alpha=0.3, label='_nolegend_')
    
    # Plot target training F1 (Phase 2 only)  
    if target_trainf1i is not None:
        target_trainf1i_sliced = target_trainf1i[1:]  # Remove first value
        x_target_trainf1i = np.array(range(len(target_trainf1i_sliced)))
        ax2.plot(x_target_trainf1i, target_trainf1i_sliced,
                alpha=0.3, color=colors['target'], linestyle=linestyles['train'], linewidth=1, label='_nolegend_')
    
    # Backward compatibility for old trainf1i
    if trainf1i is not None and base_trainf1i is None and target_trainf1i is None:
        x_trainf1i = np.array(range(len(trainf1i)))
        ax2.plot(x_trainf1i, trainf1i, color=colors['base'], linestyle=linestyles['train'],
                linewidth=1, alpha=0.3, label='_nolegend_')
    
    if testf1i is not None:
        # Plot test F1 scores (raw, low opacity) 
        x_testf1i = np.array(range(len(testf1i)))
        ax2.plot(x_testf1i, testf1i, color=colors['target'], linestyle=linestyles['test'],
                linewidth=1, alpha=0.3, label='_nolegend_')
    
    if targetf1i is not None:
        # Plot target F1 scores (raw, low opacity)
        x_targetf1i = np.array(range(len(targetf1i)))
        ax2.plot(x_targetf1i, targetf1i, color=colors['target'], linestyle=linestyles['test'],
                linewidth=1, alpha=0.3, label='_nolegend_')
    
    # Plot base training F1 moving average (Phase 1)
    if base_trainf1i is not None and len(base_trainf1i) > ma_window_size:
        base_trainf1i_ma = moving_average(base_trainf1i[1:], ma_window_size)  # Remove first value
        x_base_trainf1i_ma = np.arange(ma_window_size-1, len(base_trainf1i)-1)
        base_train_f1_label = custom_labels.get('base_train_f1', 'Base Training F1 (MA)') if custom_labels else 'Base Training F1 (MA)'
        ax2.plot(x_base_trainf1i_ma, base_trainf1i_ma, color=colors['base'], linestyle=linestyles['train'],
                linewidth=2, label=base_train_f1_label)
        
        # Mark maximum base training F1 (no marker, just annotation)
        max_idx = np.argmax(base_trainf1i_ma)
        max_val = np.max(base_trainf1i_ma)
    
    # Plot target training F1 moving average (continuous from epoch 0)
    if target_trainf1i is not None and len(target_trainf1i) > ma_window_size:
        target_trainf1i_ma = moving_average(target_trainf1i[1:], ma_window_size)  # Remove first value
        x_target_trainf1i_ma = np.arange(ma_window_size-1, len(target_trainf1i)-1)
        target_train_f1_label = custom_labels.get('target_train_f1', 'Target Training F1 (MA)') if custom_labels else 'Target Training F1 (MA)'
        ax2.plot(x_target_trainf1i_ma, target_trainf1i_ma, color=colors['target'], linestyle=linestyles['train'],
                linewidth=2, label=target_train_f1_label)
        
        # Mark maximum target training F1 (no marker, just annotation)
        max_idx = np.argmax(target_trainf1i_ma)
        max_val = np.max(target_trainf1i_ma)
    
    # Backward compatibility for old trainf1i parameter
    if trainf1i is not None and base_trainf1i is None and target_trainf1i is None and len(trainf1i) > ma_window_size:
        trainf1i_ma = moving_average(trainf1i, ma_window_size)
        x_trainf1i_ma = np.linspace(ma_window_size-1, x_axis_length-1, len(trainf1i_ma))
        train_f1_label = custom_labels.get('train_f1', 'Training F1 (MA)') if custom_labels else 'Training F1 (MA)'
        ax2.plot(x_trainf1i_ma, trainf1i_ma, color=colors['base'], linestyle=linestyles['train'],
                linewidth=2, label=train_f1_label)
        
        max_idx = np.argmax(trainf1i_ma)
        max_val = np.max(trainf1i_ma)
        

    # Plot moving average if we have enough data
    if testf1i is not None and len(testf1i) > ma_window_size:
        testf1i_ma = moving_average(testf1i, ma_window_size)
        x_testf1i_ma = np.linspace(ma_window_size-1, x_axis_length-1, len(testf1i_ma))
        # Determine test F1 label
        test_f1_label = custom_labels.get('test_f1', 'Test F1 (MA)') if custom_labels else 'Test F1 (MA)'
        ax2.plot(x_testf1i_ma, testf1i_ma, color=colors['target'], linestyle=linestyles['test'],
                linewidth=2, label=test_f1_label)
        
        # Mark maximum test F1 (no marker, just annotation)
        max_idx = np.argmax(testf1i_ma)
        max_val = np.max(testf1i_ma)
    if targetf1i is not None:
        # Plot moving average if we have enough data
        if len(targetf1i) > ma_window_size:
            targetf1i_ma = moving_average(targetf1i, ma_window_size)
            x_targetf1i_ma = np.linspace(ma_window_size-1, x_axis_length-1, len(targetf1i_ma))
            # Determine target F1 label
            target_f1_label = custom_labels.get('target_f1', 'Target Test F1 (MA)') if custom_labels else 'Target Test F1 (MA)'
            ax2.plot(x_targetf1i_ma, targetf1i_ma, color=colors['target'], linestyle=linestyles['test'],
                    linewidth=2, label=target_f1_label)
            
            # Mark maximum target F1 (no marker, just annotation)
            max_idx = np.argmax(targetf1i_ma)
            max_val = np.max(targetf1i_ma)
    
    # Plot base validation F1 scores if provided
    if basevalidationf1i is not None:
        # Plot raw base validation F1 scores with low opacity
        x_basevalidationf1i = np.array(range(len(basevalidationf1i)))
        ax2.plot(x_basevalidationf1i, basevalidationf1i, color=colors['base'], linestyle=linestyles['base_val'],
                linewidth=1, alpha=0.3, label='_nolegend_')
        
        # Plot moving average if we have enough data
        if len(basevalidationf1i) > ma_window_size:
            basevalidationf1i_ma = moving_average(basevalidationf1i, ma_window_size)
            x_basevalidationf1i_ma = np.linspace(ma_window_size-1, len(basevalidationf1i)-1, len(basevalidationf1i_ma))
            # Determine base validation F1 label
            base_val_f1_label = custom_labels.get('base_val_f1', 'Base Validation F1 (MA)') if custom_labels else 'Base Validation F1 (MA)'
            ax2.plot(x_basevalidationf1i_ma, basevalidationf1i_ma, color=colors['base'], linestyle=linestyles['base_val'],
                    linewidth=2, label=base_val_f1_label)
            
            # Mark maximum base validation F1 (no marker, just annotation)
            max_idx = np.argmax(basevalidationf1i_ma)
            max_val = np.max(basevalidationf1i_ma)
    
    # Plot target validation F1 scores if provided
    if targetvalidationf1i is not None:
        # Plot raw target validation F1 scores with low opacity
        x_targetvalidationf1i = np.array(range(len(targetvalidationf1i)))
        ax2.plot(x_targetvalidationf1i, targetvalidationf1i, color=colors['target'], linestyle=linestyles['target_val'],
                linewidth=1, alpha=0.3, label='_nolegend_')
        
        # Plot moving average if we have enough data
        if len(targetvalidationf1i) > ma_window_size:
            targetvalidationf1i_ma = moving_average(targetvalidationf1i, ma_window_size)
            x_targetvalidationf1i_ma = np.linspace(ma_window_size-1, len(targetvalidationf1i)-1, len(targetvalidationf1i_ma))
            # Determine target validation F1 label
            target_val_f1_label = custom_labels.get('target_val_f1', 'Target Validation F1 (MA)') if custom_labels else 'Target Validation F1 (MA)'
            ax2.plot(x_targetvalidationf1i_ma, targetvalidationf1i_ma, color=colors['target'], linestyle=linestyles['target_val'],
                    linewidth=2, label=target_val_f1_label)
            
            # Mark maximum target validation F1 (no marker, just annotation)
            max_idx = np.argmax(targetvalidationf1i_ma)
            max_val = np.max(targetvalidationf1i_ma)
            
    if transition_epoch is not None:
        # Add vertical line for transition epoch
        if transition_epoch > 0:
            # Add vertical line for transition epoch
            ax1.axvline(x=transition_epoch, color='gray', linestyle='--', linewidth=1, label='Transition Epoch')
            ax2.axvline(x=transition_epoch, color='gray', linestyle='--', linewidth=1, label='Transition Epoch')


    # Professional styling for F1 subplot
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_xlabel('Epochs', fontsize=12, fontweight='bold')
    ax2.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax2.set_title('F1 Scores', fontsize=14, fontweight='bold', pad=10)
    
    # Calculate dynamic y-limits based on actual F1 values to expand the view
    all_f1_values = []
    
    # Collect all F1 values that were plotted
    if trainf1i:
        all_f1_values.extend(trainf1i)
    if testf1i:
        all_f1_values.extend(testf1i)
    if targetf1i:
        all_f1_values.extend(targetf1i)
    if basevalidationf1i:
        all_f1_values.extend(basevalidationf1i)
    if targetvalidationf1i:
        all_f1_values.extend(targetvalidationf1i)
    if base_trainf1i:
        all_f1_values.extend(base_trainf1i)
    if target_trainf1i:
        all_f1_values.extend(target_trainf1i)
    
    if all_f1_values:
        min_f1 = min(all_f1_values)
        max_f1 = max(all_f1_values)
        
        # Add some padding (10% of range) and ensure we don't go below 0 or above 1
        f1_range = max_f1 - min_f1
        padding = max(0.05, f1_range * 0.1)  # At least 0.05 padding, or 10% of range
        
        y_min = max(0, min_f1 - padding)
        y_max = min(1.0, max_f1 + padding)
        
        # Ensure minimum range for visibility (at least 0.2 range)
        if y_max - y_min < 0.2:
            center = (y_min + y_max) / 2
            y_min = max(0, center - 0.1)
            y_max = min(1.0, center + 0.1)
        
        ax2.set_ylim([y_min, y_max])
        print(f"📊 F1 plot y-range: [{y_min:.3f}, {y_max:.3f}] (expanded from data range [{min_f1:.3f}, {max_f1:.3f}])")
    else:
        ax2.set_ylim([0, 1.05])  # Default range if no F1 data
    
    # Only show legend if there are labeled plots
    handles, labels = ax2.get_legend_handles_labels()
    if handles:
        ax2.legend(frameon=True, framealpha=0.9)


        
    # Main title for the whole figure
    fig.suptitle('Training Progress', fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout and save
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.12)  # Reduce space between subplots
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def load_data(raw_dataset_path, session_name, sensor_config=None, start_ns=None, stop_ns=None):
    """Load sensor data for a specific session, optionally filtered by time range"""
    try:
        # Default sensor config for backward compatibility
        if sensor_config is None:
            sensor_config = {'use_accelerometer': True, 'use_gyroscope': False}
        
        # Try exact match first
        session_path = os.path.join(raw_dataset_path, session_name.split('.')[0])
        
        df = None
        
        # Load accelerometer data if enabled
        if sensor_config.get('use_accelerometer', True):
            accelerometer_path = os.path.join(session_path, 'accelerometer_data.csv')
            accel_df = pd.read_csv(accelerometer_path)
            if 'accel_x' not in accel_df.columns:
                accel_df.rename(columns={'x': 'accel_x', 'y': 'accel_y', 'z': 'accel_z'}, inplace=True)
            df = accel_df
        
        # Load gyroscope data if enabled
        if sensor_config.get('use_gyroscope', False):
            gyroscope_path = os.path.join(session_path, 'gyroscope_data.csv')
            gyro_df = pd.read_csv(gyroscope_path)
            if 'gyro_x' not in gyro_df.columns:
                gyro_df.rename(columns={'x': 'gyro_x', 'y': 'gyro_y', 'z': 'gyro_z'}, inplace=True)
            
            if df is not None:
                # Merge accelerometer and gyroscope data using merge_asof on nearest timestamp
                df = pd.merge_asof(df.sort_values('ns_since_reboot'), 
                                 gyro_df.sort_values('ns_since_reboot'),
                                 on='ns_since_reboot', 
                                 direction='nearest')
            else:
                df = gyro_df
        
        if df is None:
            raise ValueError("No sensors enabled in sensor_config")
        
        # Filter by time range if provided
        if start_ns is not None and stop_ns is not None:
            mask = (df['ns_since_reboot'] >= start_ns) & (df['ns_since_reboot'] <= stop_ns)
            df = df[mask].copy()
            df.reset_index(drop=True, inplace=True)
        
        return df
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading data for session {session_name}: {e}")
        return None
    
def make_windowed_dataset_from_sessions(sessions, window_size, window_stride, raw_dataset_path, labeling='andrew smoking labels', sensor_config=None):
    X = []
    y = []

    # Default sensor config for backward compatibility
    if sensor_config is None:
        sensor_config = {'use_accelerometer': True, 'use_gyroscope': False}

    # Determine which columns to use based on sensor config
    sensor_columns = []
    if sensor_config.get('use_accelerometer', True):
        sensor_columns.extend(['accel_x', 'accel_y', 'accel_z'])
    if sensor_config.get('use_gyroscope', False):
        sensor_columns.extend(['gyro_x', 'gyro_y', 'gyro_z'])

    for session in sessions:
        session_name = session['session_name']
        start_ns = session.get('start_ns')
        stop_ns = session.get('stop_ns')
        bouts = [b for b in session['bouts'] if b['label'] == labeling]

        df = load_data(raw_dataset_path, session_name, sensor_config, start_ns, stop_ns)
        df = resample(df)
        
        df['label'] = 0

        for bout in bouts:
            start = bout['start']
            end = bout['end']
            df.loc[(df['ns_since_reboot'] >= start) & (df['ns_since_reboot'] <= end), 'label'] = 1

        # Ensure backward compatibility for column naming
        if 'accel_x' not in df.columns and sensor_config.get('use_accelerometer', True):
            if 'x' in df.columns:
                df.rename(columns={'x': 'accel_x', 'y': 'accel_y', 'z': 'accel_z'}, inplace=True)

        # Select only the columns we need based on sensor config
        data_columns = sensor_columns + ['label']
        missing_columns = [col for col in data_columns if col not in df.columns]
        if missing_columns:
            print(f"Warning: Missing columns {missing_columns} in session {session_name}")
            continue

        data = torch.tensor(df[data_columns].values, dtype=torch.float32)

        if data.shape[0] < window_size:
            # Zero pad the data to window size
            padding_length = window_size - data.shape[0]
            padding = torch.zeros((padding_length, data.shape[1]), dtype=torch.float32)
            data = torch.cat([data, padding], dim=0)
            print(f"Zero-padded session {session_name} from {data.shape[0] - padding_length} to {data.shape[0]} samples")

        windowed_data = data.unfold(dimension=0,size=window_size,step=window_stride)
        X.append(windowed_data[:,:-1,:])
        y.append(windowed_data[:,-1,:])

    X = torch.cat(X)
    y = (~(torch.cat(y) == 0).all(axis=1)).float()
    return X,y

def get_db_connection():
    try:
        MYSQL_CONFIG = {
            'host': os.getenv('MYSQL_HOST'),
            'user': os.getenv('MYSQL_USER'),
            'password': os.getenv('MYSQL_PASSWORD'),
            'database': os.getenv('MYSQL_DATABASE')
        }
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        return conn
    except Error as e:
        print(f"Error connecting to MySQL: {e}")

def get_project_path(project_name):
    """
    Get the project path from the database.
    """
    with db_connection() as conn:
        cursor = conn.cursor()
        query = "SELECT path FROM projects WHERE project_name = %s"
        cursor.execute(query, (project_name,))
        row = cursor.fetchone()

    if row:
        return row[0]
    else:
        raise ValueError(f"Project '{project_name}' not found in the database.")

def get_project_id(project_name):
    """Get the project ID from the database."""
    with db_connection() as conn:
        cursor = conn.cursor()
        query = "SELECT project_id FROM projects WHERE project_name = %s"
        cursor.execute(query, (project_name,))
        row = cursor.fetchone()

    if row:
        return row[0]
    else:
        raise ValueError(f"Project '{project_name}' not found in the database.")

def get_raw_dataset_path(project_name):
    """Get the raw dataset path for a project."""
    with db_connection() as conn:
        cursor = conn.cursor()
        
        # Get project_id
        project_id = get_project_id(project_name)
        
        # Get dataset_id from project_dataset_refs
        query = "SELECT dataset_id FROM project_dataset_refs WHERE project_id = %s"
        cursor.execute(query, (project_id,))
        ref_row = cursor.fetchone()
        
        if not ref_row:
            raise ValueError(f"No raw dataset reference found for project '{project_name}'")
        
        dataset_id = ref_row[0]
        
        # Get file_path from raw_datasets table
        query = "SELECT file_path FROM raw_datasets WHERE dataset_id = %s"
        cursor.execute(query, (dataset_id,))
        path_row = cursor.fetchone()
    
    if path_row and path_row[0]:
        return path_row[0]
    else:
        raise ValueError(f"No file_path found for dataset {dataset_id}")

def get_sessions_for_project(project_name):
    """Get all sessions for a project."""
    with db_connection() as conn:
        cursor = conn.cursor(dictionary=True)
        project_id = get_project_id(project_name)
        query = "SELECT * FROM sessions WHERE project_id = %s"
        cursor.execute(query, (project_id,))
        rows = cursor.fetchall()
    
    for row in rows:
        if 'bouts' in row and row['bouts']:
            try:
                row['bouts'] = json.loads(row['bouts'])
            except json.JSONDecodeError:
                row['bouts'] = []
    
    return rows
    
def get_verified_and_not_deleted_sessions(project_name, labeling):
    """
    Get sessions from the database for a specific project and labeling.
    """
    # Connect to the database and fetch sessions
    conn = get_db_connection()
    cursor = conn.cursor()
    # Execute a query
    query = f"SELECT session_name,bouts,start_ns,stop_ns FROM sessions WHERE project_id='{get_project_id(project_name)}' AND verified=1 AND (keep IS NULL OR keep != 0)"  # Replace with your actual table name
    cursor.execute(query)
    # Fetch all rows from the executed query
    rows = cursor.fetchall()
    conn.close()
    data = {'sessions': 
            [
                {
                    'session_name': row[0], 
                    'bouts': [b for b in json.loads(row[1]) if type(b) == dict and b.get('label') == labeling],
                    'bout_duration': sum((bout['end'] - bout['start']) * 1e-9 for bout in json.loads(row[1]) if type(bout) == dict and bout.get('label') == labeling),
                    'session_duration': (row[3] - row[2]) * 1e-9
                } for row in rows
            ]}

    return data

def get_participant_id(participant_code):
    """Get the participant ID from the database based on participant code."""
    with db_connection() as conn:
        cursor = conn.cursor()
        query = "SELECT participant_id FROM participants WHERE participant_code = %s"
        cursor.execute(query, (participant_code,))
        row = cursor.fetchone()

    if row:
        return row[0]
    else:
        raise ValueError(f"Participant code '{participant_code}' not found in the database.")
    
def get_participant_projects(participant_id):
    """Get all projects associated with a participant."""
    with db_connection() as conn:
        cursor = conn.cursor()
        query = "SELECT project_name FROM projects WHERE participant_id = %s"
        cursor.execute(query, (participant_id,))
        rows = cursor.fetchall()
    return [row[0] for row in rows]

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

class ResidualBlock(nn.Module):
    """RegNet-style residual block with bottleneck design for 1D time series."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResidualBlock, self).__init__()
        
        # Bottleneck design: reduce -> process -> expand
        bottleneck_channels = out_channels // 4
        
        self.conv1 = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(bottleneck_channels)
        
        self.conv2 = nn.Conv1d(bottleneck_channels, bottleneck_channels, kernel_size=kernel_size, 
                               stride=stride, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(bottleneck_channels)
        
        self.conv3 = nn.Conv1d(bottleneck_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(out_channels)
        
        # Skip connection
        self.skip = None
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = relu(self.bn1(self.conv1(x)))
        out = relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        if self.skip is not None:
            identity = self.skip(x)
            
        out += identity
        return relu(out)

class SmokingCNN(nn.Module):
    """RegNet-inspired CNN for smoking detection from time series data."""
    def __init__(self, window_size=3000, num_features=6):
        super(SmokingCNN, self).__init__()
        
        # Initial stem layer
        self.stem = nn.Sequential(
            nn.Conv1d(num_features, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )
        
        # Stage 1: 32 channels, high resolution (no pooling)
        self.stage1 = nn.Sequential(
            ResidualBlock(32, 32, kernel_size=7),
            ResidualBlock(32, 32, kernel_size=7)
        )
        
        # Stage 2: 64 channels, 2x downsampled  
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.stage2 = nn.Sequential(
            ResidualBlock(32, 64, kernel_size=5),
            ResidualBlock(64, 64, kernel_size=5),
            ResidualBlock(64, 64, kernel_size=5)
        )
        
        # Stage 3: 128 channels, 4x downsampled total
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  
        self.stage3 = nn.Sequential(
            ResidualBlock(64, 128, kernel_size=5),
            ResidualBlock(128, 128, kernel_size=5),
            ResidualBlock(128, 128, kernel_size=5)
        )
        
        # Stage 4: 256 channels, 8x downsampled total
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.stage4 = nn.Sequential(
            ResidualBlock(128, 256, kernel_size=3),
            ResidualBlock(256, 256, kernel_size=3)
        )
        
        # Global average pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(256, 1)
    
    def forward(self, x):
        x = self.stem(x)
        
        x = self.stage1(x)
        x = self.pool1(x)
        
        x = self.stage2(x)
        x = self.pool2(x)
        
        x = self.stage3(x) 
        x = self.pool3(x)
        
        x = self.stage4(x)
        
        x = self.global_pool(x)
        x = x.squeeze(-1)  # Remove sequence dimension
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x

class SimpleResidualBlock(nn.Module):
    """Lightweight residual block for MediumSmokingCNN."""
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super(SimpleResidualBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Skip connection
        self.skip = None
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.skip is not None:
            identity = self.skip(x)

        out += identity
        return relu(out)

class MediumSmokingCNN(nn.Module):
    """Medium complexity CNN with residual connections - balanced between Simple and full SmokingCNN."""
    def __init__(self, window_size=3000, num_features=6):
        super(MediumSmokingCNN, self).__init__()

        # Initial convolution
        self.stem = nn.Sequential(
            nn.Conv1d(num_features, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )

        # Residual blocks with increasing channels
        self.layer1 = SimpleResidualBlock(32, 64, kernel_size=5)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.layer2 = SimpleResidualBlock(64, 128, kernel_size=5)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.layer3 = SimpleResidualBlock(128, 256, kernel_size=3)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(256, 1)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.pool1(x)

        x = self.layer2(x)
        x = self.pool2(x)

        x = self.layer3(x)
        x = self.pool3(x)

        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.dropout(x)
        x = self.classifier(x)

        return x

class SimpleSmokingCNN(nn.Module):
    """Simple 3-layer CNN for fast training/testing."""
    def __init__(self, window_size=3000, num_features=6):
        super(SimpleSmokingCNN, self).__init__()

        self.conv1 = nn.Conv1d(num_features, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(64)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(64, 1)

    def forward(self, x):
        x = relu(self.bn1(self.conv1(x)))
        x = relu(self.bn2(self.conv2(x)))
        x = relu(self.bn3(self.conv3(x)))

        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.dropout(x)
        x = self.classifier(x)

        return x

def calculate_positive_ratio(y_tensor):
    """Calculate the ratio of positive samples in the dataset."""
    positive_count = torch.sum(y_tensor).item()
    total_count = len(y_tensor)
    return positive_count / total_count if total_count > 0 else 0.0

def init_final_layer_bias_for_imbalance(model, positive_ratio):
    """Initialize final layer bias based on class distribution (Karpathy technique).
    
    This eliminates the 'hockey stick' training curve by initializing the final layer
    bias to the log-odds of the positive class ratio.
    
    Args:
        model: The neural network model
        positive_ratio: Fraction of positive samples in training data (0-1)
    """
    if positive_ratio <= 0 or positive_ratio >= 1:
        print(f"Warning: Invalid positive ratio {positive_ratio:.4f}, skipping bias initialization")
        return
    
    # Calculate log-odds for bias initialization
    bias_init = np.log(positive_ratio / (1 - positive_ratio))
    
    # Find the final linear layer in the model
    final_layer = None
    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Linear):
            final_layer = module
            break
    
    if final_layer is not None:
        final_layer.bias.data.fill_(bias_init)
        print(f"✅ Initialized final layer bias to {bias_init:.4f} (positive ratio: {positive_ratio:.4f})")
    else:
        print("Warning: Could not find final linear layer for bias initialization")

def get_projects_from_participant_codes(participant_codes):
    projects = []
    for participant_code in participant_codes:
        participant_id = get_participant_id(participant_code)
        participant_projects = get_participant_projects(participant_id)
        if len(participant_projects) == 0:
            print(f"No projects found for participant {participant_code}.")
            continue
        print(f"Participant {participant_code} has projects: {participant_projects}")
        projects.extend(participant_projects)
    return projects

def generate_dataset_summary(config, session_data, train_test_splits, X_train, y_train, X_test, y_test, save_dir):
    """
    Generate comprehensive dataset summary with statistics and save as CSV and text report.
    
    Args:
        config: Configuration dictionary
        session_data: Dictionary mapping project_name -> list of sessions
        train_test_splits: Dictionary mapping project_name -> (train_sessions, test_sessions)
        X_train, y_train: Training data tensors
        X_test, y_test: Test data tensors
        save_dir: Directory to save summary files
    """
    import pandas as pd
    from datetime import datetime
    import os
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Collect session-level statistics
    session_stats = []
    project_stats = []
    
    for project_name, sessions in session_data.items():
        train_sessions, test_sessions = train_test_splits[project_name]
        train_session_names = {s['session_name'] for s in train_sessions}
        
        # Project-level aggregates
        total_sessions = len(sessions)
        total_duration = 0
        total_smoking_duration = 0
        smoking_sessions = 0
        
        for session in sessions:
            session_name = session['session_name']
            split_type = 'train' if session_name in train_session_names else 'test'
            
            # Calculate session duration (convert from nanoseconds to minutes)
            if session.get('start_ns') and session.get('stop_ns'):
                duration_minutes = (session['stop_ns'] - session['start_ns']) * 1e-9 / 60
            else:
                duration_minutes = 0
            
            # Calculate smoking duration in this session
            smoking_duration_minutes = 0
            smoking_bouts = [b for b in session.get('bouts', []) if b.get('label') == config['dataset']['labeling']]
            for bout in smoking_bouts:
                smoking_duration_minutes += (bout['end'] - bout['start']) * 1e-9 / 60
            
            if smoking_duration_minutes > 0:
                smoking_sessions += 1
            
            total_duration += duration_minutes
            total_smoking_duration += smoking_duration_minutes
            
            session_stats.append({
                'project_name': project_name,
                'session_name': session_name,
                'split': split_type,
                'duration_minutes': round(duration_minutes, 2),
                'smoking_duration_minutes': round(smoking_duration_minutes, 2),
                'smoking_percentage': round((smoking_duration_minutes / duration_minutes * 100) if duration_minutes > 0 else 0, 2),
                'num_smoking_bouts': len(smoking_bouts),
                'keep_status': session.get('keep', 1)
            })
        
        project_stats.append({
            'project_name': project_name,
            'total_sessions': total_sessions,
            'train_sessions': len(train_sessions),
            'test_sessions': len(test_sessions),
            'total_duration_hours': round(total_duration / 60, 2),
            'smoking_duration_hours': round(total_smoking_duration / 60, 2),
            'smoking_percentage': round((total_smoking_duration / total_duration * 100) if total_duration > 0 else 0, 2),
            'sessions_with_smoking': smoking_sessions,
            'smoking_session_percentage': round((smoking_sessions / total_sessions * 100) if total_sessions > 0 else 0, 2)
        })
    
    # Convert to DataFrames
    session_df = pd.DataFrame(session_stats)
    project_df = pd.DataFrame(project_stats)
    
    # Calculate windowed data statistics
    train_samples = len(X_train)
    test_samples = len(X_test)
    total_samples = train_samples + test_samples
    
    train_positive = int(torch.sum(y_train).item())
    train_negative = int(train_samples - train_positive)
    test_positive = int(torch.sum(y_test).item())
    test_negative = int(test_samples - test_positive)
    
    # Create summary statistics
    summary_stats = {
        'dataset_info': {
            'generation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'config_file': 'config.yaml',
            'experiment_name': config['experiment']['name'],
            'labeling_type': config['dataset']['labeling'],
            'window_size_samples': config['dataset']['window_size'],
            'window_size_seconds': config['dataset']['window_size_seconds'],
            'window_stride_samples': config['dataset']['window_stride'],
            'window_stride_seconds': config['dataset']['window_stride_seconds'],
            'sampling_frequency_hz': config['dataset']['fs'],
            'test_split_ratio': config['dataset']['test_size']
        },
        'project_summary': {
            'total_projects': len(project_stats),
            'total_sessions': session_df.shape[0],
            'total_duration_hours': round(session_df['duration_minutes'].sum() / 60, 2),
            'total_smoking_hours': round(session_df['smoking_duration_minutes'].sum() / 60, 2),
            'overall_smoking_percentage': round((session_df['smoking_duration_minutes'].sum() / session_df['duration_minutes'].sum() * 100), 2)
        },
        'split_summary': {
            'train_sessions': len(session_df[session_df['split'] == 'train']),
            'test_sessions': len(session_df[session_df['split'] == 'test']),
            'train_duration_hours': round(session_df[session_df['split'] == 'train']['duration_minutes'].sum() / 60, 2),
            'test_duration_hours': round(session_df[session_df['split'] == 'test']['duration_minutes'].sum() / 60, 2)
        },
        'windowed_data_summary': {
            'total_windows': total_samples,
            'train_windows': train_samples,
            'test_windows': test_samples,
            'train_percentage': round((train_samples / total_samples * 100), 2),
            'test_percentage': round((test_samples / total_samples * 100), 2),
            'train_positive_windows': train_positive,
            'train_negative_windows': train_negative,
            'test_positive_windows': test_positive,
            'test_negative_windows': test_negative,
            'train_positive_percentage': round((train_positive / train_samples * 100), 2),
            'train_negative_percentage': round((train_negative / train_samples * 100), 2),
            'test_positive_percentage': round((test_positive / test_samples * 100), 2),
            'test_negative_percentage': round((test_negative / test_samples * 100), 2),
            'overall_positive_percentage': round(((train_positive + test_positive) / total_samples * 100), 2),
            'class_balance_ratio': round((train_negative + test_negative) / (train_positive + test_positive), 2)
        }
    }
    
    # Save session-level details as CSV
    session_csv_path = os.path.join(save_dir, f'dataset_sessions_{timestamp}.csv')
    session_df.to_csv(session_csv_path, index=False)
    
    # Save project-level summary as CSV
    project_csv_path = os.path.join(save_dir, f'dataset_projects_{timestamp}.csv')
    project_df.to_csv(project_csv_path, index=False)
    
    # Save comprehensive text summary
    summary_txt_path = os.path.join(save_dir, f'dataset_summary_{timestamp}.txt')
    with open(summary_txt_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DATASET SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("DATASET CONFIGURATION\n")
        f.write("-"*40 + "\n")
        for key, value in summary_stats['dataset_info'].items():
            f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        
        f.write("\nPROJECT OVERVIEW\n")
        f.write("-"*40 + "\n")
        for key, value in summary_stats['project_summary'].items():
            f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        
        f.write("\nTRAIN/TEST SPLIT OVERVIEW\n")
        f.write("-"*40 + "\n")
        for key, value in summary_stats['split_summary'].items():
            f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        
        f.write("\nWINDOWED DATA STATISTICS\n")
        f.write("-"*40 + "\n")
        for key, value in summary_stats['windowed_data_summary'].items():
            f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        
        f.write("\nPROJECT DETAILS\n")
        f.write("-"*40 + "\n")
        f.write(project_df.to_string(index=False))
        
        f.write("\n\nCLASS DISTRIBUTION BY SPLIT\n")
        f.write("-"*40 + "\n")
        f.write(f"Training Set:\n")
        f.write(f"  - Negative (No Smoking): {train_negative:,} windows ({summary_stats['windowed_data_summary']['train_negative_percentage']:.1f}%)\n")
        f.write(f"  - Positive (Smoking): {train_positive:,} windows ({summary_stats['windowed_data_summary']['train_positive_percentage']:.1f}%)\n")
        f.write(f"\nTest Set:\n")
        f.write(f"  - Negative (No Smoking): {test_negative:,} windows ({summary_stats['windowed_data_summary']['test_negative_percentage']:.1f}%)\n")
        f.write(f"  - Positive (Smoking): {test_positive:,} windows ({summary_stats['windowed_data_summary']['test_positive_percentage']:.1f}%)\n")
        
        f.write(f"\nOverall Class Balance:\n")
        f.write(f"  - Negative:Positive Ratio = {summary_stats['windowed_data_summary']['class_balance_ratio']:.2f}:1\n")
        
    print(f"\n📊 Dataset summary generated:")
    print(f"   📄 Session details: {session_csv_path}")
    print(f"   📈 Project summary: {project_csv_path}")
    print(f"   📋 Full report: {summary_txt_path}")
    
    return {
        'session_df': session_df,
        'project_df': project_df,
        'summary_stats': summary_stats,
        'files': {
            'sessions_csv': session_csv_path,
            'projects_csv': project_csv_path,
            'summary_txt': summary_txt_path
        }
    }


# =============================================================================
# ADVANCED CUSTOMIZATION TECHNIQUES
# =============================================================================

class EWCRegularizer:
    """Elastic Weight Consolidation (EWC) for preventing catastrophic forgetting."""
    
    def __init__(self, model, dataloader, device, num_samples=1000):
        self.model = model
        self.device = device
        self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = self._compute_fisher_information(dataloader, num_samples)
    
    def _compute_fisher_information(self, dataloader, num_samples):
        """Compute Fisher Information Matrix for EWC regularization."""
        fisher = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                fisher[n] = torch.zeros_like(p)
        
        self.model.eval()
        criterion = torch.nn.BCEWithLogitsLoss()
        
        samples_processed = 0
        for X, y in dataloader:
            if samples_processed >= num_samples:
                break
                
            X, y = X.to(self.device), y.to(self.device).float()
            
            # Forward pass
            outputs = self.model(X).squeeze()
            loss = criterion(outputs, y)
            
            # Backward pass to get gradients
            self.model.zero_grad()
            loss.backward()
            
            # Accumulate squared gradients (Fisher Information)
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data ** 2
            
            samples_processed += len(X)
        
        # Average Fisher Information
        for n in fisher:
            fisher[n] /= samples_processed
        
        return fisher
    
    def penalty(self, model):
        """Compute EWC penalty loss."""
        loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.fisher:
                loss += (self.fisher[n] * (p - self.params[n]) ** 2).sum()
        return loss


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


def coral_loss(source_features, target_features):
    """Compute CORAL (Correlation Alignment) loss for domain adaptation."""
    # Compute covariance matrices
    source_cov = torch.cov(source_features.T)
    target_cov = torch.cov(target_features.T)
    
    # Compute Frobenius norm of difference
    loss = torch.norm(source_cov - target_cov, 'fro') ** 2
    
    # Normalize by feature dimension
    d = source_features.shape[1]
    loss = loss / (4 * d * d)
    
    return loss


def contrastive_loss(features, labels, temperature=0.1):
    """Compute supervised contrastive loss."""
    batch_size = features.shape[0]
    
    # Normalize features
    features = torch.nn.functional.normalize(features, dim=1)
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(features, features.T) / temperature
    
    # Create mask for positive pairs (same label)
    labels = labels.unsqueeze(1)
    mask = torch.eq(labels, labels.T).float()
    
    # Remove self-similarity
    mask = mask - torch.eye(batch_size, device=features.device)
    
    # Compute contrastive loss
    exp_sim = torch.exp(similarity_matrix)
    sum_exp_sim = torch.sum(exp_sim, dim=1, keepdim=True)
    
    log_prob = similarity_matrix - torch.log(sum_exp_sim)
    mean_log_prob_pos = torch.sum(mask * log_prob, dim=1) / (torch.sum(mask, dim=1) + 1e-8)
    
    loss = -mean_log_prob_pos.mean()
    
    return loss


class LayerwiseFinetuner:
    """Handles layer-wise fine-tuning and gradual unfreezing."""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.layer_groups = self._create_layer_groups()
        self.frozen_groups = set(range(len(self.layer_groups)))
    
    def _create_layer_groups(self):
        """Create layer groups for gradual unfreezing."""
        layer_groups = []
        
        # Group 1: Early conv layers
        if hasattr(self.model, 'conv1'):
            layer_groups.append(['conv1'])
        
        # Group 2: Middle conv layers  
        conv_layers = []
        for name, _ in self.model.named_modules():
            if 'conv' in name and name != 'conv1':
                conv_layers.append(name)
        if conv_layers:
            layer_groups.append(conv_layers)
        
        # Group 3: Final layers (classifier)
        final_layers = []
        for name, _ in self.model.named_modules():
            if any(keyword in name for keyword in ['fc', 'classifier', 'linear']):
                final_layers.append(name)
        if final_layers:
            layer_groups.append(final_layers)
        
        return layer_groups
    
    def freeze_all_except_classifier(self):
        """Freeze all parameters except the classifier."""
        for name, param in self.model.named_parameters():
            # Keep classifier unfrozen
            if any(keyword in name for keyword in ['fc', 'classifier', 'linear']):
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    def unfreeze_group(self, group_idx):
        """Unfreeze a specific layer group."""
        if group_idx < len(self.layer_groups):
            group = self.layer_groups[group_idx]
            for layer_name in group:
                for name, param in self.model.named_parameters():
                    if layer_name in name:
                        param.requires_grad = True
            self.frozen_groups.discard(group_idx)
    
    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.model.parameters():
            param.requires_grad = True
        self.frozen_groups.clear()
    
    def get_layerwise_optimizer(self, base_lr, lr_multiplier=0.1):
        """Create optimizer with different learning rates for different layer groups."""
        param_groups = []
        
        # Classifier parameters (highest LR)
        classifier_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and any(keyword in name for keyword in ['fc', 'classifier', 'linear']):
                classifier_params.append(param)
        
        if classifier_params:
            param_groups.append({'params': classifier_params, 'lr': base_lr})
        
        # Other unfrozen parameters (lower LR)
        other_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and not any(keyword in name for keyword in ['fc', 'classifier', 'linear']):
                other_params.append(param)
        
        if other_params:
            param_groups.append({'params': other_params, 'lr': base_lr * lr_multiplier})
        
        return torch.optim.AdamW(param_groups, weight_decay=1e-4)


class EnsemblePredictor:
    """Ensemble predictor combining base and target models."""
    
    def __init__(self, base_model_path, target_model_path, alpha=0.7):
        self.alpha = alpha  # Weight for base model
        self.base_model_path = base_model_path
        self.target_model_path = target_model_path
        self.base_model = None
        self.target_model = None
    
    def load_models(self, model_class, model_kwargs):
        """Load both base and target models."""
        # Load base model
        self.base_model = model_class(**model_kwargs)
        self.base_model.load_state_dict(torch.load(self.base_model_path))
        self.base_model.eval()
        
        # Load target model
        self.target_model = model_class(**model_kwargs)
        self.target_model.load_state_dict(torch.load(self.target_model_path))
        self.target_model.eval()
    
    def predict(self, X):
        """Make ensemble predictions."""
        with torch.no_grad():
            base_pred = self.base_model(X)
            target_pred = self.target_model(X)
            
            # Weighted ensemble
            ensemble_pred = self.alpha * base_pred + (1 - self.alpha) * target_pred
            
        return ensemble_pred


def extract_features_for_coral(model, dataloader, device, layer_name='features'):
    """Extract features from a specific layer for CORAL loss computation."""
    features = []
    labels = []
    
    # Register hook to extract features
    feature_outputs = {}
    
    def hook_fn(module, input, output):
        feature_outputs['features'] = output
    
    # Find the layer to hook into
    hook_handles = []
    for name, module in model.named_modules():
        if layer_name in name:
            handle = module.register_forward_hook(hook_fn)
            hook_handles.append(handle)
            break
    
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            # Forward pass
            _ = model(X)
            
            # Extract features
            if 'features' in feature_outputs:
                batch_features = feature_outputs['features']
                # Flatten if needed
                if len(batch_features.shape) > 2:
                    batch_features = batch_features.view(batch_features.size(0), -1)
                
                features.append(batch_features.cpu())
                labels.append(y.cpu())
    
    # Remove hooks
    for handle in hook_handles:
        handle.remove()
    
    if features:
        return torch.cat(features, dim=0), torch.cat(labels, dim=0)
    else:
        return None, None