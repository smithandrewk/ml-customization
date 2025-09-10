"""
Two-phase leave-one-participant-out training script with base training + customization
"""
import os
import argparse
from time import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, WeightedRandomSampler
import numpy as np

from sklearn.metrics import f1_score

from utils import load_config, get_experiment_dir, get_next_experiment_dir, plot_training_progress, SmokingCNN, SimpleSmokingCNN, calculate_positive_ratio, init_final_layer_bias_for_imbalance, create_stratified_combined_dataloader, EWCRegularizer, TimeSeriesAugmenter, coral_loss, contrastive_loss, LayerwiseFinetuner, EnsemblePredictor, extract_features_for_coral
from dotenv import load_dotenv

load_dotenv()

# Parse command line arguments
parser = argparse.ArgumentParser(description='Two-phase customization training for smoking detection')
parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
parser.add_argument('--dataset_dir', type=str, required=True, help='Directory containing participant-specific train/test files')
parser.add_argument('--target_participant', type=str, required=True, help='Participant to customize for (hold out for phase 2)')
parser.add_argument('--experiment_suffix', type=str, help='Suffix for experiment name (default: custom_{target_participant})')
parser.add_argument('--model', type=str, default='full', choices=['full', 'simple'], help='Model architecture: full (RegNet-style) or simple (3-layer for testing)')
parser.add_argument('--early_stopping_metric', type=str, default='f1', choices=['f1', 'loss'], help='Metric for early stopping: f1 (maximize F1) or loss (minimize loss)')
parser.add_argument('--target_weight', type=float, help='Weight multiplier for target samples in Phase 2 (overrides config, default: 1.0 = equal weighting)')
parser.add_argument('--use_stratified_sampling', action='store_true', help='Use stratified sampling instead of WeightedRandomSampler to prevent extreme batch compositions')

# Advanced technique flags (for ablation studies)
parser.add_argument('--use_layerwise_finetuning', action='store_true', help='Enable layer-wise fine-tuning')
parser.add_argument('--use_gradual_unfreezing', action='store_true', help='Enable gradual unfreezing')
parser.add_argument('--use_ewc', action='store_true', help='Enable Elastic Weight Consolidation (EWC)')
parser.add_argument('--use_augmentation', action='store_true', help='Enable time-series data augmentation')
parser.add_argument('--use_coral', action='store_true', help='Enable CORAL domain adaptation')
parser.add_argument('--use_ensemble', action='store_true', help='Enable ensemble approach')
parser.add_argument('--use_contrastive', action='store_true', help='Enable contrastive learning')
args = parser.parse_args()

# Load configuration
config = load_config(args.config)

# Override target weight if specified via command line
if args.target_weight is not None:
    config['training']['target_weight_multiplier'] = args.target_weight
    print(f"Overriding target weight multiplier from config: {args.target_weight}")

# Override advanced technique flags if specified via command line
advanced_config = config['training'].get('advanced_techniques', {})
if args.use_layerwise_finetuning:
    advanced_config['use_layerwise_finetuning'] = True
    print("🔧 Enabling layer-wise fine-tuning via command line")
if args.use_gradual_unfreezing:
    advanced_config['use_gradual_unfreezing'] = True
    print("🔧 Enabling gradual unfreezing via command line")
if args.use_ewc:
    advanced_config['use_ewc'] = True
    print("🔧 Enabling EWC regularization via command line")
if args.use_augmentation:
    advanced_config['use_augmentation'] = True
    print("🔧 Enabling data augmentation via command line")
if args.use_coral:
    advanced_config['use_coral'] = True
    print("🔧 Enabling CORAL domain adaptation via command line")
if args.use_ensemble:
    advanced_config['use_ensemble'] = True
    print("🔧 Enabling ensemble approach via command line")
if args.use_contrastive:
    advanced_config['use_contrastive'] = True
    print("🔧 Enabling contrastive learning via command line")

target_weight_multiplier = config['training']['target_weight_multiplier']

# Validate target weight multiplier
if target_weight_multiplier < 0:
    raise ValueError(f"target_weight_multiplier must be >= 0, got {target_weight_multiplier}")

# Provide warnings and guidance for edge cases
if target_weight_multiplier == 0.0:
    print("⚠️  WARNING: target_weight=0.0 will exclude target data entirely")
    print("   This effectively runs Phase 1 training with Phase 2 validation")
    print("   Consider using target_weight >= 0.1 for meaningful target representation")
elif target_weight_multiplier < 0.1:
    print(f"⚠️  WARNING: Very low target_weight={target_weight_multiplier:.2f} may provide minimal target representation")
    print("   Consider target_weight >= 0.1 for meaningful customization")
elif target_weight_multiplier > 10.0:
    print(f"⚠️  WARNING: Very high target_weight={target_weight_multiplier:.1f} may overwhelm base knowledge")
    print("   Consider target_weight <= 10.0 for balanced training")

# Extract dataset name from dataset_dir for new experiment naming
# e.g., "data/001_tonmoy_60s" -> "tonmoy_60s"
dataset_name = os.path.basename(args.dataset_dir).split('_', 1)[1] if '_' in os.path.basename(args.dataset_dir) else os.path.basename(args.dataset_dir)
experiment_dir = get_next_experiment_dir(f"{dataset_name}_custom_{args.target_participant}", "custom")

def load_participant_data(dataset_dir, participants, split='train', is_base_participants=False):
    """Load and return separate datasets for each participant."""
    datasets = []
    participant_info = {}
    
    for participant in participants:
        file_path = f'{dataset_dir}/{participant}_{split}.pt'
        if os.path.exists(file_path):
            X, y = torch.load(file_path)
            
            # For base participants doing training, combine train + val for larger training set
            if split == 'train' and is_base_participants:
                val_path = f'{dataset_dir}/{participant}_val.pt'
                if os.path.exists(val_path):
                    X_val, y_val = torch.load(val_path)
                    X = torch.cat([X, X_val], dim=0)
                    y = torch.cat([y, y_val], dim=0)
                    print(f"Combined {participant}_train.pt + {participant}_val.pt for base training: {len(X):,} samples")
                else:
                    print(f"Loaded {participant}_{split}.pt: {len(X):,} samples")
            else:
                print(f"Loaded {participant}_{split}.pt: {len(X):,} samples")
                
            datasets.append(TensorDataset(X, y))
            participant_info[participant] = len(X)
        else:
            print(f"Warning: {file_path} not found")
    
    return datasets, participant_info

def create_combined_dataloader(datasets, batch_size, shuffle=True):
    """Create a dataloader from multiple datasets."""
    if datasets:
        combined_dataset = ConcatDataset(datasets)
        return DataLoader(combined_dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        raise ValueError("No datasets provided")

def create_weighted_combined_dataloader(base_datasets, target_datasets, target_weight_multiplier, batch_size):
    """Create a weighted dataloader combining base and target datasets.
    
    Args:
        base_datasets: List of base participant datasets
        target_datasets: List of target participant datasets  
        target_weight_multiplier: Weight multiplier for target samples (1.0 = equal weighting, 0.0 = exclude target)
        batch_size: Batch size for the dataloader
    
    Returns:
        DataLoader with weighted sampling or regular sampling for edge cases
    """
    if not base_datasets and not target_datasets:
        raise ValueError("No datasets provided")
    
    # Special case: target_weight = 0 means exclude target data entirely
    if target_weight_multiplier == 0.0:
        if not base_datasets:
            raise ValueError("Cannot create dataloader with target_weight=0 and no base datasets")
        print("⚠️  target_weight=0.0: Excluding target data entirely (Phase 1 only)")
        combined_dataset = ConcatDataset(base_datasets)
        return DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
    
    # Special case: no target datasets provided
    if not target_datasets:
        combined_dataset = ConcatDataset(base_datasets)
        return DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
    
    # Normal case: weighted sampling with both base and target data
    all_datasets = base_datasets + target_datasets
    combined_dataset = ConcatDataset(all_datasets)
    
    # Calculate weights for each sample
    weights = []
    
    # Add base weights (weight = 1.0)
    for dataset in base_datasets:
        weights.extend([1.0] * len(dataset))
    
    # Add target weights (weight = target_weight_multiplier)
    for dataset in target_datasets:
        weights.extend([target_weight_multiplier] * len(dataset))
    
    # Create weighted sampler
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    
    # Create dataloader with weighted sampling
    return DataLoader(combined_dataset, batch_size=batch_size, sampler=sampler)

# Get all available participants from dataset directory
available_participants = []
for file in os.listdir(args.dataset_dir):
    if file.endswith('_train.pt'):
        participant = file.replace('_train.pt', '')
        available_participants.append(participant)

print(f"Available participants: {available_participants}")

# Validate target participant
if args.target_participant not in available_participants:
    raise ValueError(f"Target participant '{args.target_participant}' not found in dataset directory")

# Define base participants (all except target)
base_participants = [p for p in available_participants if p != args.target_participant]
print(f"Base participants (N-1): {base_participants}")
print(f"Target participant: {args.target_participant}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============================================================================
# PHASE 1: BASE TRAINING (N-1 participants)
# =============================================================================
print("\n" + "="*60)
print("PHASE 1: BASE TRAINING")
print("="*60)

# Load base training data (N-1 participants) - combines train + val for larger training set
base_train_datasets, base_train_info = load_participant_data(args.dataset_dir, base_participants, 'train', is_base_participants=True)
base_val_datasets, base_val_info = load_participant_data(args.dataset_dir, base_participants, 'test')

# Also load target participant test and validation data for evaluation throughout both phases
target_test_datasets, target_test_info = load_participant_data(args.dataset_dir, [args.target_participant], 'test')
target_val_datasets, target_val_info = load_participant_data(args.dataset_dir, [args.target_participant], 'val')

# Load target training data for continuous evaluation (not used for actual training in Phase 1)
target_train_datasets_eval, target_train_info_eval = load_participant_data(args.dataset_dir, [args.target_participant], 'train')
target_trainloader_eval = create_combined_dataloader(target_train_datasets_eval, config['training']['batch_size'], shuffle=False)

# Create dataloaders with clear names
base_trainloader = create_combined_dataloader(base_train_datasets, config['training']['batch_size'], shuffle=True)
base_valloader = create_combined_dataloader(base_val_datasets, config['training']['batch_size'], shuffle=False)
target_testloader = create_combined_dataloader(target_test_datasets, config['training']['batch_size'], shuffle=False)
target_valloader = create_combined_dataloader(target_val_datasets, config['training']['batch_size'], shuffle=False)

print(f"Base training samples (train+val): {sum(base_train_info.values()):,}")
print(f"Base validation samples (test): {sum(base_val_info.values()):,}")
print(f"Target test samples: {sum(target_test_info.values()):,}")

# Initialize model based on selection
if args.model == 'simple':
    print("Using SimpleSmokingCNN (3-layer) for fast training/testing")
    model = SimpleSmokingCNN(
        window_size=config['model']['window_size'], 
        num_features=config['model']['num_features']
    )
else:
    print("Using full SmokingCNN (RegNet-style) architecture")
    model = SmokingCNN(
        window_size=config['model']['window_size'], 
        num_features=config['model']['num_features']
    )

# Calculate positive ratio from base training data for bias initialization
base_y_labels = []
for dataset in base_train_datasets:
    for _, y in dataset:
        if y.dim() == 0:  # scalar
            base_y_labels.append(y.unsqueeze(0))
        else:
            base_y_labels.append(y)
base_y_tensor = torch.cat(base_y_labels)
base_positive_ratio = calculate_positive_ratio(base_y_tensor)
print(f"Base training positive ratio: {base_positive_ratio:.4f} ({torch.sum(base_y_tensor).item():,}/{len(base_y_tensor):,})")

# Initialize final layer bias based on class distribution
init_final_layer_bias_for_imbalance(model, base_positive_ratio)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(
    model.parameters(), 
    lr=config['training']['learning_rate'], 
    weight_decay=config['training']['weight_decay']
)

model.to(device)

# Phase 1 training metrics
base_trainlossi = []
base_vallossi = []
base_trainf1i = []
base_valf1i = []

# Target test metrics for Phase 1 (to track target performance during base training)
target_testlossi_phase1 = []
target_testf1i_phase1 = []
# Target validation metrics for Phase 1 (to track target validation baseline)
target_vallossi_phase1 = []
target_valf1i_phase1 = []

# Continuous training evaluation metrics (evaluate both throughout both phases)
base_train_continuous_lossi = []  # Base training loss evaluated continuously
base_train_continuous_f1i = []   # Base training F1 evaluated continuously  
target_train_continuous_lossi = [] # Target training loss evaluated continuously
target_train_continuous_f1i = []  # Target training F1 evaluated continuously

# Phase 1 early stopping variables
base_n_epochs_without_improvement = 0
if args.early_stopping_metric == 'f1':
    base_best_metric = 0.0  # Start with 0 for F1 (maximize)
    base_metric_name = 'Base Val F1'
else:  # loss
    base_best_metric = float('inf')  # Start with infinity for loss (minimize)  
    base_metric_name = 'Base Val Loss'
base_best_epoch = -1
base_patience = config['training'].get('base_patience', config['training'].get('patience', 10))

print(f"Starting Phase 1 training with patience={base_patience}")

# Initial evaluation before training starts (epoch -1, will be plotted as epoch 0)
print("Performing initial evaluation before training...")
model.eval()

# Evaluate initial performance on both base and target training data
with torch.no_grad():
    # Base training evaluation
    base_train_preds_init = []
    base_train_labels_init = []
    base_train_losses_init = []
    
    for Xi, yi in base_trainloader:
        Xi, yi = Xi.to(device), yi.to(device).float()
        outputs = model(Xi).squeeze()
        loss = criterion(outputs, yi)
        
        base_train_preds_init.append((outputs.sigmoid() > 0.5).float())
        base_train_labels_init.append(yi)
        base_train_losses_init.append(loss.item())
    
    base_train_preds_init = torch.cat(base_train_preds_init).cpu()
    base_train_labels_init = torch.cat(base_train_labels_init).cpu()
    base_train_loss_init = np.mean(base_train_losses_init)
    base_train_continuous_lossi.append(base_train_loss_init)
    base_train_continuous_f1i.append(f1_score(base_train_labels_init, base_train_preds_init, average='macro', zero_division=0))
    
    # Target training evaluation
    target_train_preds_init = []
    target_train_labels_init = []
    target_train_losses_init = []
    
    for Xi, yi in target_trainloader_eval:
        Xi, yi = Xi.to(device), yi.to(device).float()
        outputs = model(Xi).squeeze()
        loss = criterion(outputs, yi)
        
        target_train_preds_init.append((outputs.sigmoid() > 0.5).float())
        target_train_labels_init.append(yi)
        target_train_losses_init.append(loss.item())
    
    target_train_preds_init = torch.cat(target_train_preds_init).cpu()
    target_train_labels_init = torch.cat(target_train_labels_init).cpu()
    target_train_loss_init = np.mean(target_train_losses_init)
    target_train_continuous_lossi.append(target_train_loss_init)
    target_train_continuous_f1i.append(f1_score(target_train_labels_init, target_train_preds_init, average='macro', zero_division=0))

print(f"Initial base training loss: {base_train_loss_init:.4f}, F1: {base_train_continuous_f1i[-1]:.4f}")
print(f"Initial target training loss: {target_train_loss_init:.4f}, F1: {target_train_continuous_f1i[-1]:.4f}")

for epoch in range(config['training']['num_epochs']):
    epoch_start_time = time()
    model.train()

    train_preds = []
    train_labels = []
    train_losses = []

    for Xi, yi in base_trainloader:
        Xi, yi = Xi.to(device), yi.to(device).float()
        optimizer.zero_grad()
        outputs = model(Xi).squeeze()
        loss = criterion(outputs, yi)
        loss.backward()
        
        # Add gradient clipping to Phase 1 as well
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        # Minimize GPU→CPU transfers - keep on GPU until end of epoch
        train_preds.append((outputs.sigmoid() > 0.5).float())
        train_labels.append(yi)
        train_losses.append(loss.item())

    # Single GPU→CPU transfer per epoch
    train_preds = torch.cat(train_preds).cpu()
    train_labels = torch.cat(train_labels).cpu()
    train_loss = np.mean(train_losses)
    base_trainlossi.append(train_loss)
    base_trainf1i.append(f1_score(train_labels, train_preds, average='macro', zero_division=0))

    # Base validation evaluation
    model.eval()
    val_preds = []
    val_labels = []
    val_losses = []

    with torch.no_grad():
        for Xi, yi in base_valloader:
            Xi, yi = Xi.to(device), yi.to(device).float()
            outputs = model(Xi).squeeze()
            loss = criterion(outputs, yi)

            val_preds.append((outputs.sigmoid() > 0.5).float())
            val_labels.append(yi)
            val_losses.append(loss.item())

    val_preds = torch.cat(val_preds).cpu()
    val_labels = torch.cat(val_labels).cpu()
    val_loss = np.mean(val_losses)
    base_vallossi.append(val_loss)
    base_valf1i.append(f1_score(val_labels, val_preds, average='macro', zero_division=0))

    # Also evaluate on target test set during Phase 1 (to see how base training affects target performance)
    target_test_preds = []
    target_test_labels = []
    target_test_losses = []

    with torch.no_grad():
        for Xi, yi in target_testloader:
            Xi, yi = Xi.to(device), yi.to(device).float()
            outputs = model(Xi).squeeze()
            loss = criterion(outputs, yi)

            target_test_preds.append((outputs.sigmoid() > 0.5).float())
            target_test_labels.append(yi)
            target_test_losses.append(loss.item())

    target_test_preds = torch.cat(target_test_preds).cpu()
    target_test_labels = torch.cat(target_test_labels).cpu()
    target_test_loss = np.mean(target_test_losses)
    target_testlossi_phase1.append(target_test_loss)
    target_testf1i_phase1.append(f1_score(target_test_labels, target_test_preds, average='macro', zero_division=0))

    # Also evaluate on target validation set during Phase 1 (to track target validation baseline)
    target_val_preds = []
    target_val_labels = []
    target_val_losses = []

    with torch.no_grad():
        for Xi, yi in target_valloader:
            Xi, yi = Xi.to(device), yi.to(device).float()
            outputs = model(Xi).squeeze()
            loss = criterion(outputs, yi)

            target_val_preds.append((outputs.sigmoid() > 0.5).float())
            target_val_labels.append(yi)
            target_val_losses.append(loss.item())

    target_val_preds = torch.cat(target_val_preds).cpu()
    target_val_labels = torch.cat(target_val_labels).cpu()
    target_val_loss = np.mean(target_val_losses)
    target_vallossi_phase1.append(target_val_loss)
    target_valf1i_phase1.append(f1_score(target_val_labels, target_val_preds, average='macro', zero_division=0))

    # Continuous training evaluation: Evaluate current model on both base and target training data
    model.eval()
    
    # Evaluate on base training data (continuous)
    base_train_preds = []
    base_train_labels = []
    base_train_losses = []
    
    with torch.no_grad():
        for Xi, yi in base_trainloader:
            Xi, yi = Xi.to(device), yi.to(device).float()
            outputs = model(Xi).squeeze()
            loss = criterion(outputs, yi)
            
            base_train_preds.append((outputs.sigmoid() > 0.5).float())
            base_train_labels.append(yi)
            base_train_losses.append(loss.item())
    
    base_train_preds = torch.cat(base_train_preds).cpu()
    base_train_labels = torch.cat(base_train_labels).cpu()
    base_train_loss = np.mean(base_train_losses)
    base_train_continuous_lossi.append(base_train_loss)
    base_train_continuous_f1i.append(f1_score(base_train_labels, base_train_preds, average='macro', zero_division=0))
    
    # Evaluate on target training data (continuous) - will be poor in Phase 1 since no target data used for training
    target_train_preds = []
    target_train_labels = []
    target_train_losses = []
    
    # Use the target trainloader created at the beginning
    
    with torch.no_grad():
        for Xi, yi in target_trainloader_eval:
            Xi, yi = Xi.to(device), yi.to(device).float()
            outputs = model(Xi).squeeze()
            loss = criterion(outputs, yi)
            
            target_train_preds.append((outputs.sigmoid() > 0.5).float())
            target_train_labels.append(yi)
            target_train_losses.append(loss.item())
    
    target_train_preds = torch.cat(target_train_preds).cpu()
    target_train_labels = torch.cat(target_train_labels).cpu()
    target_train_loss = np.mean(target_train_losses)
    target_train_continuous_lossi.append(target_train_loss)
    target_train_continuous_f1i.append(f1_score(target_train_labels, target_train_preds, average='macro', zero_division=0))

    # Plot training progress every epoch for Phase 1 with 4-line format
    if epoch % 1 == 0:
        # Create current arrays for Phase 1 plotting (no Phase 2 data yet)
        current_basevalidationlossi = base_vallossi
        current_basevalidationf1i = base_valf1i
        current_targetvalidationlossi = target_vallossi_phase1
        current_targetvalidationf1i = target_valf1i_phase1
        
        # Define descriptive labels for Phase 1
        phase1_labels = {
            'train': 'Training Log Loss: Phase 1 (Base Training)',  # Legacy
            'target': 'Target Test Log Loss (Continuous Evaluation)',
            'base_val': 'Base Validation Log Loss (Continuous Evaluation)', 
            'target_val': 'Target Validation Log Loss (Continuous Evaluation)',
            'train_f1': 'Training F1: Phase 1 (Base Training)',  # Legacy
            'target_f1': 'Target Test F1 (Continuous Evaluation)',
            'base_val_f1': 'Base Validation F1 (Continuous Evaluation)',
            'target_val_f1': 'Target Validation F1 (Continuous Evaluation)',
            # New separate training labels
            'base_train': 'Base Training Log Loss (Phase 1)',
            'target_train': 'Target Training Log Loss (Phase 2)',  # Not shown in Phase 1
            'base_train_f1': 'Base Training F1 (Phase 1)',
            'target_train_f1': 'Target Training F1 (Phase 2)'  # Not shown in Phase 1
        }
        
        plot_training_progress(
            trainlossi=base_trainlossi,  # Legacy - keep for backward compatibility
            testlossi=None,  # Legacy parameter - not used
            targetlossi=target_testlossi_phase1,  # Target test loss during base training
            basevalidationlossi=current_basevalidationlossi,  # Base validation loss
            targetvalidationlossi=current_targetvalidationlossi,  # Target validation loss
            trainf1i=base_trainf1i,  # Legacy - keep for backward compatibility
            testf1i=None,  # Legacy parameter - not used
            targetf1i=target_testf1i_phase1,  # Target test F1 during base training
            basevalidationf1i=current_basevalidationf1i,  # Base validation F1
            targetvalidationf1i=current_targetvalidationf1i,  # Target validation F1
            # New separate training parameters for Phase 1 (continuous evaluation)
            base_trainlossi=base_train_continuous_lossi,  # Base training loss (continuous evaluation)
            target_trainlossi=target_train_continuous_lossi,  # Target training loss (continuous evaluation)
            base_trainf1i=base_train_continuous_f1i,  # Base training F1 (continuous evaluation)
            target_trainf1i=target_train_continuous_f1i,  # Target training F1 (continuous evaluation)
            ma_window_size=config['visualization']['ma_window_size'],
            save_path=f'{experiment_dir}/training_progress.{config["visualization"]["plot_format"]}',
            transition_epoch=None,  # No transition in Phase 1
            custom_labels=phase1_labels
        )

    # Phase 1 early stopping (based on chosen metric)
    if args.early_stopping_metric == 'f1':
        current_metric = base_valf1i[-1]
        improved = current_metric > base_best_metric
    else:  # loss
        current_metric = base_vallossi[-1]
        improved = current_metric < base_best_metric
    
    if improved:
        base_best_metric = current_metric
        base_best_epoch = epoch
        torch.save(model.state_dict(), f'{experiment_dir}/base_model.pt')
        print(f"Phase 1 Epoch {epoch}: New best base model saved with {base_metric_name}: {base_best_metric:.4f}")
        base_n_epochs_without_improvement = 0
    else:
        print(f"Phase 1 Epoch {epoch}: No improvement ({current_metric:.4f} vs {base_best_metric:.4f})")
        base_n_epochs_without_improvement += 1

    if base_n_epochs_without_improvement >= base_patience:
        print(f"Phase 1 early stopping triggered after {base_n_epochs_without_improvement} epochs without improvement.")
        break
        
    # Skip intermediate plotting - will show complete plot at end

    epoch_time = time() - epoch_start_time
    print(f"Phase 1 Epoch {epoch} completed in {epoch_time:.2f}s. "
          f"Train Loss: {train_loss:.4f}, Base Val Loss: {val_loss:.4f}, Target Test Loss: {target_test_loss:.4f}, "
          f"Train F1: {base_trainf1i[-1]:.4f}, Base Val F1: {base_valf1i[-1]:.4f}, Target Test F1: {target_testf1i_phase1[-1]:.4f}. "
          f"Epochs without improvement: {base_n_epochs_without_improvement}")

print(f"\nPhase 1 completed! Best {base_metric_name}: {base_best_metric:.4f}")

# Save Phase 1 metrics
base_metrics = {
    'train_loss': base_trainlossi,
    'val_loss': base_vallossi,
    'target_test_loss': target_testlossi_phase1,
    'train_f1': base_trainf1i,
    'val_f1': base_valf1i,
    'target_test_f1': target_testf1i_phase1,
    'best_val_metric': base_best_metric,
    'base_participants': base_participants,
    'base_train_samples': sum(base_train_info.values()),
    'base_val_samples': sum(base_val_info.values())
}
torch.save(base_metrics, f'{experiment_dir}/base_metrics.pt')

# =============================================================================
# PHASE 2: CUSTOMIZATION TRAINING (Base + Target)
# =============================================================================
print("\n" + "="*60)
print("PHASE 2: CUSTOMIZATION TRAINING")
print("="*60)

# Load the best base model
model.load_state_dict(torch.load(f'{experiment_dir}/base_model.pt'))
print("Loaded best base model for customization")

# =============================================================================
# INITIALIZE ADVANCED CUSTOMIZATION TECHNIQUES
# =============================================================================

# Initialize EWC regularizer if enabled
ewc_regularizer = None
if advanced_config.get('use_ewc', False):
    print("🧠 Initializing EWC regularizer...")
    ewc_samples = advanced_config.get('fisher_samples', 1000)
    ewc_regularizer = EWCRegularizer(model, base_trainloader, device, num_samples=ewc_samples)
    print(f"   Fisher Information computed using {ewc_samples} samples")

# Initialize data augmenter if enabled
augmenter = None
if advanced_config.get('use_augmentation', False):
    print("🔄 Initializing time-series data augmenter...")
    aug_config = advanced_config.get('augmentation', {})
    augmenter = TimeSeriesAugmenter(aug_config)
    print(f"   Augmentation probability: {aug_config.get('augmentation_probability', 0.5)}")

# Initialize layer-wise fine-tuner if enabled
layerwise_finetuner = None
if advanced_config.get('use_layerwise_finetuning', False) or advanced_config.get('use_gradual_unfreezing', False):
    print("🎯 Initializing layer-wise fine-tuner...")
    layerwise_finetuner = LayerwiseFinetuner(model, advanced_config)
    print(f"   Layer groups identified: {len(layerwise_finetuner.layer_groups)}")
    
    # Start with classifier-only training if layer-wise fine-tuning is enabled
    if advanced_config.get('use_layerwise_finetuning', False):
        layerwise_finetuner.freeze_all_except_classifier()
        print("   🔒 Frozen all layers except classifier for initial fine-tuning")

# Print enabled techniques summary
enabled_techniques = []
if advanced_config.get('use_layerwise_finetuning', False):
    enabled_techniques.append("Layer-wise Fine-tuning")
if advanced_config.get('use_gradual_unfreezing', False):
    enabled_techniques.append("Gradual Unfreezing")
if advanced_config.get('use_ewc', False):
    enabled_techniques.append("EWC Regularization")
if advanced_config.get('use_augmentation', False):
    enabled_techniques.append("Data Augmentation")
if advanced_config.get('use_coral', False):
    enabled_techniques.append("CORAL Domain Adaptation")
if advanced_config.get('use_contrastive', False):
    enabled_techniques.append("Contrastive Learning")
if advanced_config.get('use_ensemble', False):
    enabled_techniques.append("Ensemble Approach")

if enabled_techniques:
    print(f"\n🚀 Advanced techniques enabled: {', '.join(enabled_techniques)}")
else:
    print("\n📍 Using baseline customization approach (no advanced techniques)")

# Get lambda values for loss combinations
ewc_lambda = advanced_config.get('ewc_lambda', 1000.0)
coral_lambda = advanced_config.get('coral_lambda', 1.0)
contrastive_lambda = advanced_config.get('contrastive_lambda', 0.1)

# Load target participant data
target_train_datasets, target_train_info = load_participant_data(args.dataset_dir, [args.target_participant], 'train')
target_val_datasets, target_val_info = load_participant_data(args.dataset_dir, [args.target_participant], 'val')
# target_test_datasets already loaded in Phase 1

# FIX 3: Create combined training data with stratified sampling option
if args.use_stratified_sampling:
    print("🎯 Using stratified sampling to prevent extreme batch compositions")
    combined_trainloader = create_stratified_combined_dataloader(
        base_train_datasets, 
        target_train_datasets, 
        target_weight_multiplier, 
        config['training']['batch_size']
    )
else:
    print("⚠️  Using WeightedRandomSampler - may cause extreme batch compositions")
    combined_trainloader = create_weighted_combined_dataloader(
        base_train_datasets, 
        target_train_datasets, 
        target_weight_multiplier, 
        config['training']['batch_size']
    )

# Create validation dataloaders
target_valloader = create_combined_dataloader(target_val_datasets, config['training']['batch_size'], shuffle=False)
# Keep target_testloader from Phase 1 (already created)

# Calculate effective target representation with weighting
base_samples = sum(base_train_info.values())
target_samples = sum(target_train_info.values())

if target_weight_multiplier == 0.0:
    # Special case: target data excluded entirely
    effective_target_percentage = 0.0
    total_weight = base_samples * 1.0  # Only base samples count
else:
    # Normal case: weighted representation
    total_weight = base_samples * 1.0 + target_samples * target_weight_multiplier
    effective_target_percentage = (target_samples * target_weight_multiplier / total_weight) * 100

if target_weight_multiplier == 0.0:
    print(f"Training samples (target excluded): {base_samples:,}")
    print(f"  - Base participants: {base_samples:,} (weight: 1.0)")
    print(f"  - Target participant: {target_samples:,} (weight: {target_weight_multiplier:.1f} → EXCLUDED)")
    print(f"  - Effective target representation: {effective_target_percentage:.1f}% (no target data used)")
else:
    print(f"Combined training samples: {base_samples + target_samples:,}")
    print(f"  - Base participants: {base_samples:,} (weight: 1.0)")
    print(f"  - Target participant: {target_samples:,} (weight: {target_weight_multiplier:.1f})")
    print(f"  - Effective target representation: {effective_target_percentage:.1f}%")
print(f"Target validation samples: {sum(target_val_info.values()):,}")
print(f"Target test samples: {sum(target_test_info.values()):,}")

# Calculate positive ratio from combined training data for Phase 2 bias initialization
combined_y_labels = []
# Collect labels from base datasets
for dataset in base_train_datasets:
    for _, y in dataset:
        if y.dim() == 0:  # scalar
            combined_y_labels.append(y.unsqueeze(0))
        else:
            combined_y_labels.append(y)
# Collect labels from target datasets
for dataset in target_train_datasets:
    for _, y in dataset:
        if y.dim() == 0:  # scalar
            combined_y_labels.append(y.unsqueeze(0))
        else:
            combined_y_labels.append(y)
combined_y_tensor = torch.cat(combined_y_labels)
combined_positive_ratio = calculate_positive_ratio(combined_y_tensor)
print(f"Combined training positive ratio: {combined_positive_ratio:.4f} ({torch.sum(combined_y_tensor).item():,}/{len(combined_y_tensor):,})")

# Note: NOT re-initializing final layer bias to preserve learned classification boundary from Phase 1
print(f"Keeping trained final layer bias from Phase 1 (combined positive ratio: {combined_positive_ratio:.4f})")

# Create new optimizer for customization phase
custom_lr = config['training'].get('custom_learning_rate', config['training']['learning_rate'] * 0.1)

# Use layer-wise optimizer if layer-wise fine-tuning is enabled
if layerwise_finetuner is not None:
    lr_multiplier = advanced_config.get('layerwise_lr_multiplier', 0.1)
    optimizer = layerwise_finetuner.get_layerwise_optimizer(custom_lr, lr_multiplier)
    print(f"   📊 Using layer-wise optimizer (classifier LR: {custom_lr}, other layers LR: {custom_lr * lr_multiplier})")
else:
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=custom_lr,
        weight_decay=config['training']['weight_decay']
    )
    print(f"   📊 Using standard optimizer (LR: {custom_lr})")

# Phase 2 training metrics
custom_trainlossi = []
custom_target_vallossi = []  # Target validation for early stopping
custom_target_testlossi = []  # Target test for final reporting
custom_trainf1i = []
custom_target_valf1i = []  # Target validation F1 for early stopping
custom_target_testf1i = []  # Target test F1 for final reporting
# Base validation metrics for Phase 2 (for continuous tracking)
base_vallossi_phase2 = []
base_valf1i_phase2 = []

# Phase 2 early stopping variables
custom_n_epochs_without_improvement = 0
if args.early_stopping_metric == 'f1':
    custom_best_metric = 0.0  # Start with 0 for F1 (maximize)
    custom_metric_name = 'Target Val F1'
else:  # loss
    custom_best_metric = float('inf')  # Start with infinity for loss (minimize)
    custom_metric_name = 'Target Val Loss'
custom_best_epoch = -1
custom_patience = config['training'].get('custom_patience', config['training'].get('patience', 10) // 2)

print(f"Starting Phase 2 customization with LR={custom_lr}, patience={custom_patience}")

for epoch in range(config['training']['num_epochs']):
    epoch_start_time = time()
    model.train()

    train_preds = []
    train_labels = []
    train_losses = []
    extreme_batch_count = 0
    
    # Handle gradual unfreezing
    if layerwise_finetuner is not None and advanced_config.get('use_gradual_unfreezing', False):
        unfreeze_schedule = advanced_config.get('unfreeze_schedule', [5, 10, 15])
        for unfreeze_epoch in unfreeze_schedule:
            if epoch == unfreeze_epoch:
                group_idx = unfreeze_schedule.index(unfreeze_epoch)
                if group_idx < len(layerwise_finetuner.layer_groups):
                    layerwise_finetuner.unfreeze_group(group_idx)
                    print(f"🔓 Unfroze layer group {group_idx} at epoch {epoch}")
                    
                    # Update optimizer with new unfrozen parameters
                    lr_multiplier = advanced_config.get('layerwise_lr_multiplier', 0.1)
                    optimizer = layerwise_finetuner.get_layerwise_optimizer(custom_lr, lr_multiplier)
    
    # Handle layer-wise fine-tuning transition
    if (layerwise_finetuner is not None and 
        advanced_config.get('use_layerwise_finetuning', False) and 
        epoch == advanced_config.get('layerwise_classifier_epochs', 5)):
        layerwise_finetuner.unfreeze_all()
        lr_multiplier = advanced_config.get('layerwise_lr_multiplier', 0.1)
        optimizer = layerwise_finetuner.get_layerwise_optimizer(custom_lr, lr_multiplier)
        print(f"🔓 Unfroze all layers after {epoch} classifier-only epochs")

    for batch_idx, (Xi, yi) in enumerate(combined_trainloader):
        Xi, yi = Xi.to(device), yi.to(device).float()
        
        # Apply data augmentation if enabled
        if augmenter is not None:
            Xi, yi = augmenter.augment(Xi, yi)
        
        # FIX 1: Batch composition logging to detect extreme batches
        batch_positive_ratio = yi.mean().item()
        current_loss_before = None
        if batch_positive_ratio > 0.9 or batch_positive_ratio < 0.1:
            extreme_batch_count += 1
            # Calculate loss before training step for extreme batches
            with torch.no_grad():
                outputs_pre = model(Xi).squeeze()
                current_loss_before = criterion(outputs_pre, yi).item()
            if epoch % 5 == 0 or extreme_batch_count <= 5:  # Log first few or every 5th epoch
                print(f"    Extreme batch {batch_idx}: {batch_positive_ratio:.3f} positive ratio, pre-step loss: {current_loss_before:.4f}")
        
        optimizer.zero_grad()
        outputs = model(Xi).squeeze()
        
        # Compute base BCE loss
        bce_loss = criterion(outputs, yi)
        total_loss = bce_loss
        
        # Add EWC regularization if enabled
        if ewc_regularizer is not None:
            ewc_loss = ewc_regularizer.penalty(model)
            total_loss += ewc_lambda * ewc_loss
        
        # Add CORAL loss if enabled (computed every few batches for efficiency)
        if advanced_config.get('use_coral', False) and batch_idx % 10 == 0:
            # Extract base and target features for CORAL
            with torch.no_grad():
                # Get a batch of base data
                try:
                    base_batch = next(iter(base_trainloader))
                    base_X = base_batch[0][:len(Xi)].to(device)  # Match batch size
                    
                    # Extract features from both domains
                    base_features, _ = extract_features_for_coral(model, [(base_X, torch.zeros(len(base_X)))], device)
                    target_features, _ = extract_features_for_coral(model, [(Xi, yi)], device)
                    
                    if base_features is not None and target_features is not None:
                        coral_loss_val = coral_loss(base_features.to(device), target_features.to(device))
                        total_loss += coral_lambda * coral_loss_val
                except:
                    pass  # Skip CORAL if extraction fails
        
        # Add contrastive loss if enabled
        if advanced_config.get('use_contrastive', False):
            # Extract features for contrastive learning
            try:
                features, _ = extract_features_for_coral(model, [(Xi, yi)], device)
                if features is not None:
                    contrastive_loss_val = contrastive_loss(features.to(device), yi)
                    total_loss += contrastive_lambda * contrastive_loss_val
            except:
                pass  # Skip contrastive if extraction fails
        
        total_loss.backward()
        
        # FIX 2: Add gradient clipping to prevent gradient explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        # Log extreme batch post-step loss
        if current_loss_before is not None and (epoch % 5 == 0 or extreme_batch_count <= 5):
            print(f"    Post-step loss: {total_loss.item():.4f}, change: {total_loss.item() - current_loss_before:+.4f}")

        # Minimize GPU→CPU transfers - keep on GPU until end of epoch
        train_preds.append((outputs.sigmoid() > 0.5).float())
        train_labels.append(yi)
        train_losses.append(total_loss.item())

    # Single GPU→CPU transfer per epoch
    train_preds = torch.cat(train_preds).cpu()
    train_labels = torch.cat(train_labels).cpu()
    train_loss = np.mean(train_losses)
    custom_trainlossi.append(train_loss)
    custom_trainf1i.append(f1_score(train_labels, train_preds, average='macro', zero_division=0))
    
    # Summary of extreme batches for this epoch
    if extreme_batch_count > 0:
        total_batches = len(combined_trainloader)
        extreme_percentage = (extreme_batch_count / total_batches) * 100
        print(f"  🚨 Extreme batches this epoch: {extreme_batch_count}/{total_batches} ({extreme_percentage:.1f}%)")

    # Evaluation on both target validation and target test
    model.eval()
    
    # Target validation evaluation (for early stopping)
    target_val_preds = []
    target_val_labels = []
    target_val_losses = []
    
    with torch.no_grad():
        for Xi, yi in target_valloader:
            Xi, yi = Xi.to(device), yi.to(device).float()
            outputs = model(Xi).squeeze()
            loss = criterion(outputs, yi)

            target_val_preds.append((outputs.sigmoid() > 0.5).float())
            target_val_labels.append(yi)
            target_val_losses.append(loss.item())

    target_val_preds = torch.cat(target_val_preds).cpu()
    target_val_labels = torch.cat(target_val_labels).cpu()
    target_val_loss = np.mean(target_val_losses)
    custom_target_vallossi.append(target_val_loss)
    custom_target_valf1i.append(f1_score(target_val_labels, target_val_preds, average='macro', zero_division=0))

    # Target test evaluation (for final reporting only)
    target_test_preds = []
    target_test_labels = []
    target_test_losses = []

    with torch.no_grad():
        for Xi, yi in target_testloader:
            Xi, yi = Xi.to(device), yi.to(device).float()
            outputs = model(Xi).squeeze()
            loss = criterion(outputs, yi)

            target_test_preds.append((outputs.sigmoid() > 0.5).float())
            target_test_labels.append(yi)
            target_test_losses.append(loss.item())

    target_test_preds = torch.cat(target_test_preds).cpu()
    target_test_labels = torch.cat(target_test_labels).cpu()
    target_test_loss = np.mean(target_test_losses)
    custom_target_testlossi.append(target_test_loss)
    custom_target_testf1i.append(f1_score(target_test_labels, target_test_preds, average='macro', zero_division=0))

    # Base validation evaluation during Phase 2 (for continuous tracking)
    base_val_preds = []
    base_val_labels = []
    base_val_losses = []

    with torch.no_grad():
        for Xi, yi in base_valloader:
            Xi, yi = Xi.to(device), yi.to(device).float()
            outputs = model(Xi).squeeze()
            loss = criterion(outputs, yi)

            base_val_preds.append((outputs.sigmoid() > 0.5).float())
            base_val_labels.append(yi)
            base_val_losses.append(loss.item())

    base_val_preds = torch.cat(base_val_preds).cpu()
    base_val_labels = torch.cat(base_val_labels).cpu()
    base_val_loss = np.mean(base_val_losses)
    base_vallossi_phase2.append(base_val_loss)
    base_valf1i_phase2.append(f1_score(base_val_labels, base_val_preds, average='macro', zero_division=0))

    # Continuous training evaluation in Phase 2: Evaluate current model on both base and target training data
    model.eval()
    
    # Evaluate on base training data (continuous) - should be similar to Phase 1 end since base data unchanged
    base_train_preds_p2 = []
    base_train_labels_p2 = []
    base_train_losses_p2 = []
    
    with torch.no_grad():
        for Xi, yi in base_trainloader:
            Xi, yi = Xi.to(device), yi.to(device).float()
            outputs = model(Xi).squeeze()
            loss = criterion(outputs, yi)
            
            base_train_preds_p2.append((outputs.sigmoid() > 0.5).float())
            base_train_labels_p2.append(yi)
            base_train_losses_p2.append(loss.item())
    
    base_train_preds_p2 = torch.cat(base_train_preds_p2).cpu()
    base_train_labels_p2 = torch.cat(base_train_labels_p2).cpu()
    base_train_loss_p2 = np.mean(base_train_losses_p2)
    base_train_continuous_lossi.append(base_train_loss_p2)
    base_train_continuous_f1i.append(f1_score(base_train_labels_p2, base_train_preds_p2, average='macro', zero_division=0))
    
    # Evaluate on target training data (continuous) - should improve in Phase 2 since target data now used for training
    target_train_preds_p2 = []
    target_train_labels_p2 = []
    target_train_losses_p2 = []
    
    with torch.no_grad():
        for Xi, yi in target_trainloader_eval:
            Xi, yi = Xi.to(device), yi.to(device).float()
            outputs = model(Xi).squeeze()
            loss = criterion(outputs, yi)
            
            target_train_preds_p2.append((outputs.sigmoid() > 0.5).float())
            target_train_labels_p2.append(yi)
            target_train_losses_p2.append(loss.item())
    
    target_train_preds_p2 = torch.cat(target_train_preds_p2).cpu()
    target_train_labels_p2 = torch.cat(target_train_labels_p2).cpu()
    target_train_loss_p2 = np.mean(target_train_losses_p2)
    target_train_continuous_lossi.append(target_train_loss_p2)
    target_train_continuous_f1i.append(f1_score(target_train_labels_p2, target_train_preds_p2, average='macro', zero_division=0))

    # Plot training progress every epoch for Phase 2 with 4-line format
    if epoch % 1 == 0:
        # Create combined arrays for Phase 2 plotting (includes both phases)
        combined_trainlossi_current = base_trainlossi + custom_trainlossi
        combined_targetlossi_current = target_testlossi_phase1 + custom_target_testlossi
        combined_basevalidationlossi_current = base_vallossi + base_vallossi_phase2
        combined_targetvalidationlossi_current = target_vallossi_phase1 + custom_target_vallossi
        combined_trainf1i_current = base_trainf1i + custom_trainf1i
        combined_targetf1i_current = target_testf1i_phase1 + custom_target_testf1i
        combined_basevalidationf1i_current = base_valf1i + base_valf1i_phase2
        combined_targetvalidationf1i_current = target_valf1i_phase1 + custom_target_valf1i
        current_transition_epoch = len(base_trainf1i) - 1
        
        # Define descriptive labels for Phase 2
        phase2_labels = {
            'train': 'Training Log Loss: Phase 1 (Base) → Phase 2 (Target)',  # Legacy
            'target': 'Target Test Log Loss (Continuous Evaluation)',
            'base_val': 'Base Validation Log Loss (Continuous Evaluation)', 
            'target_val': 'Target Validation Log Loss (Continuous Evaluation)',
            'train_f1': 'Training F1: Phase 1 (Base) → Phase 2 (Target)',  # Legacy
            'target_f1': 'Target Test F1 (Continuous Evaluation)',
            'base_val_f1': 'Base Validation F1 (Continuous Evaluation)',
            'target_val_f1': 'Target Validation F1 (Continuous Evaluation)',
            # New separate training labels
            'base_train': 'Base Training Log Loss (Phase 1)',
            'target_train': 'Target Training Log Loss (Phase 2)',
            'base_train_f1': 'Base Training F1 (Phase 1)',
            'target_train_f1': 'Target Training F1 (Phase 2)'
        }
        
        plot_training_progress(
            trainlossi=combined_trainlossi_current,  # Legacy - keep for backward compatibility
            testlossi=None,  # Legacy parameter - not used
            targetlossi=combined_targetlossi_current,  # Continuous target test
            basevalidationlossi=combined_basevalidationlossi_current,  # Continuous base validation
            targetvalidationlossi=combined_targetvalidationlossi_current,  # Continuous target validation
            trainf1i=combined_trainf1i_current,  # Legacy - keep for backward compatibility
            testf1i=None,  # Legacy parameter - not used
            targetf1i=combined_targetf1i_current,  # Continuous target test F1
            basevalidationf1i=combined_basevalidationf1i_current,  # Continuous base validation F1
            targetvalidationf1i=combined_targetvalidationf1i_current,  # Continuous target validation F1
            # New separate training parameters for Phase 2 (continuous evaluation)
            base_trainlossi=base_train_continuous_lossi,  # Base training loss (continuous evaluation)
            target_trainlossi=target_train_continuous_lossi,  # Target training loss (continuous evaluation)
            base_trainf1i=base_train_continuous_f1i,  # Base training F1 (continuous evaluation)
            target_trainf1i=target_train_continuous_f1i,  # Target training F1 (continuous evaluation)
            ma_window_size=config['visualization']['ma_window_size'],
            save_path=f'{experiment_dir}/training_progress.{config["visualization"]["plot_format"]}',
            transition_epoch=current_transition_epoch,
            custom_labels=phase2_labels
        )

    # Phase 2 early stopping (based on chosen metric)
    if args.early_stopping_metric == 'f1':
        current_metric = custom_target_valf1i[-1]
        improved = current_metric > custom_best_metric
    else:  # loss
        current_metric = custom_target_vallossi[-1]
        improved = current_metric < custom_best_metric
    
    if improved:
        custom_best_metric = current_metric
        custom_best_epoch = len(base_trainf1i) + epoch  # Adjust for combined epoch numbering
        torch.save(model.state_dict(), f'{experiment_dir}/customized_model.pt')
        print(f"Phase 2 Epoch {epoch}: New best customized model saved with {custom_metric_name}: {custom_best_metric:.4f}")
        custom_n_epochs_without_improvement = 0
    else:
        print(f"Phase 2 Epoch {epoch}: No improvement ({current_metric:.4f} vs {custom_best_metric:.4f})")
        custom_n_epochs_without_improvement += 1

    if custom_n_epochs_without_improvement >= custom_patience:
        print(f"Phase 2 early stopping triggered after {custom_n_epochs_without_improvement} epochs without improvement.")
        break
        
    # Skip intermediate plotting - will show complete plot at end

    epoch_time = time() - epoch_start_time
    print(f"Phase 2 Epoch {epoch} completed in {epoch_time:.2f}s. "
          f"Train Loss: {train_loss:.4f}, Target Val Loss: {target_val_loss:.4f}, Target Test Loss: {target_test_loss:.4f}, "
          f"Train F1: {custom_trainf1i[-1]:.4f}, Target Val F1: {custom_target_valf1i[-1]:.4f}, Target Test F1: {custom_target_testf1i[-1]:.4f}. "
          f"Epochs without improvement: {custom_n_epochs_without_improvement}")

print(f"\nPhase 2 completed! Best {custom_metric_name}: {custom_best_metric:.4f}")

# =============================================================================
# ENSEMBLE EVALUATION (if enabled)
# =============================================================================
if advanced_config.get('use_ensemble', False):
    print("\n" + "="*60)
    print("ENSEMBLE EVALUATION")
    print("="*60)
    
    ensemble_alpha = advanced_config.get('ensemble_alpha', 0.7)
    ensemble_predictor = EnsemblePredictor(
        f'{experiment_dir}/base_model.pt',
        f'{experiment_dir}/customized_model.pt',
        alpha=ensemble_alpha
    )
    
    # Initialize ensemble with model architecture
    if args.model == 'simple':
        model_class = SimpleSmokingCNN
    else:
        model_class = SmokingCNN
    
    model_kwargs = {
        'window_size': config['model']['window_size'],
        'num_features': config['model']['num_features']
    }
    
    ensemble_predictor.load_models(model_class, model_kwargs)
    ensemble_predictor.base_model.to(device)
    ensemble_predictor.target_model.to(device)
    
    # Evaluate ensemble on target test set
    ensemble_test_preds = []
    ensemble_test_labels = []
    
    with torch.no_grad():
        for Xi, yi in target_testloader:
            Xi, yi = Xi.to(device), yi.to(device).float()
            ensemble_outputs = ensemble_predictor.predict(Xi).squeeze()
            
            ensemble_test_preds.append((ensemble_outputs.sigmoid() > 0.5).float())
            ensemble_test_labels.append(yi)
    
    ensemble_test_preds = torch.cat(ensemble_test_preds).cpu()
    ensemble_test_labels = torch.cat(ensemble_test_labels).cpu()
    ensemble_test_f1 = f1_score(ensemble_test_labels, ensemble_test_preds, average='macro', zero_division=0)
    
    print(f"Ensemble Test F1 (α={ensemble_alpha}): {ensemble_test_f1:.4f}")
    print(f"  • Base model weight: {ensemble_alpha:.1f}")
    print(f"  • Target model weight: {1-ensemble_alpha:.1f}")
    
    # Store ensemble results
    custom_metrics['ensemble_test_f1'] = ensemble_test_f1
    custom_metrics['ensemble_alpha'] = ensemble_alpha

# =============================================================================
# FINAL MODEL EVALUATION (Using Moving Averages)
# =============================================================================
print("\n" + "="*60)
print("FINAL MODEL EVALUATION (Using Moving Averages)")
print("="*60)

from utils import moving_average

# Get moving average window size from config
ma_window_size = config['visualization']['ma_window_size']

# Calculate moving averages for target test F1 throughout both phases
combined_target_testf1i = target_testf1i_phase1 + custom_target_testf1i

# Extract moving average F1 at the best model epochs
if len(combined_target_testf1i) > ma_window_size:
    target_testf1i_ma = moving_average(combined_target_testf1i, ma_window_size)
    
    # Map epoch indices to moving average indices
    # Moving average starts at index (ma_window_size - 1)
    def epoch_to_ma_index(epoch):
        return max(0, min(epoch - (ma_window_size - 1), len(target_testf1i_ma) - 1))
    
    # Get moving average F1 at best model epochs
    base_model_ma_index = epoch_to_ma_index(base_best_epoch)
    custom_model_ma_index = epoch_to_ma_index(custom_best_epoch)
    
    base_model_test_f1_ma = target_testf1i_ma[base_model_ma_index]
    custom_model_test_f1_ma = target_testf1i_ma[custom_model_ma_index]
    
    print(f"Phase 1 best model target test F1 (MA): {base_model_test_f1_ma:.4f} (epoch {base_best_epoch})")
    print(f"Phase 2 best model target test F1 (MA): {custom_model_test_f1_ma:.4f} (epoch {custom_best_epoch})")
    
    # Calculate improvements using moving averages
    absolute_improvement_ma = custom_model_test_f1_ma - base_model_test_f1_ma
    percentage_improvement_ma = (absolute_improvement_ma / base_model_test_f1_ma * 100) if base_model_test_f1_ma > 0 else 0.0
    
    print(f"\nPerformance Improvement (Moving Average):")
    print(f"  • Absolute improvement: {absolute_improvement_ma:+.4f}")
    print(f"  • Percentage improvement: {percentage_improvement_ma:+.2f}%")
    
    # For backward compatibility, also store the moving average values
    base_model_test_f1 = base_model_test_f1_ma
    custom_model_test_f1 = custom_model_test_f1_ma  
    absolute_improvement = absolute_improvement_ma
    percentage_improvement = percentage_improvement_ma
    
else:
    # Fallback to raw values if not enough data for moving average
    print("Warning: Not enough epochs for moving average, using raw values")
    base_model_test_f1 = target_testf1i_phase1[base_best_epoch] if base_best_epoch < len(target_testf1i_phase1) else 0.0
    custom_epoch_in_phase2 = custom_best_epoch - len(target_testf1i_phase1)
    custom_model_test_f1 = custom_target_testf1i[custom_epoch_in_phase2] if custom_epoch_in_phase2 < len(custom_target_testf1i) else 0.0
    
    absolute_improvement = custom_model_test_f1 - base_model_test_f1
    percentage_improvement = (absolute_improvement / base_model_test_f1 * 100) if base_model_test_f1 > 0 else 0.0
    
    print(f"Phase 1 best model target test F1 (raw): {base_model_test_f1:.4f}")
    print(f"Phase 2 best model target test F1 (raw): {custom_model_test_f1:.4f}")
    print(f"Absolute improvement: {absolute_improvement:+.4f}")
    print(f"Percentage improvement: {percentage_improvement:+.2f}%")

# Save Phase 2 metrics
custom_metrics = {
    'train_loss': custom_trainlossi,
    'target_val_loss': custom_target_vallossi,
    'target_test_loss': custom_target_testlossi,
    'train_f1': custom_trainf1i,
    'target_val_f1': custom_target_valf1i,
    'target_test_f1': custom_target_testf1i,
    'base_best_metric': base_best_metric,  # Best metric value from Phase 1
    'custom_best_metric': custom_best_metric,  # Best metric value from Phase 2  
    'base_model_test_f1': base_model_test_f1,  # Moving average F1 at best Phase 1 epoch
    'custom_model_test_f1': custom_model_test_f1,  # Moving average F1 at best Phase 2 epoch
    'absolute_improvement': absolute_improvement,  # Based on moving averages
    'percentage_improvement': percentage_improvement,  # Based on moving averages
    'base_best_epoch': base_best_epoch,
    'custom_best_epoch': custom_best_epoch,
    'ma_window_size': ma_window_size,
    'evaluation_method': 'moving_average',  # Flag to indicate method used
    'early_stopping_metric': args.early_stopping_metric,  # 'f1' or 'loss'
    'base_metric_name': base_metric_name,
    'custom_metric_name': custom_metric_name,
    'target_participant': args.target_participant,
    'combined_train_samples': sum(base_train_info.values()) + sum(target_train_info.values()),
    'target_val_samples': sum(target_val_info.values()),
    'target_test_samples': sum(target_test_info.values()),
    'custom_learning_rate': custom_lr,
    'target_weight_multiplier': target_weight_multiplier,
    'effective_target_percentage': effective_target_percentage,
    # Advanced techniques used (for ablation analysis)
    'advanced_techniques_used': enabled_techniques,
    'use_layerwise_finetuning': advanced_config.get('use_layerwise_finetuning', False),
    'use_gradual_unfreezing': advanced_config.get('use_gradual_unfreezing', False),
    'use_ewc': advanced_config.get('use_ewc', False),
    'use_augmentation': advanced_config.get('use_augmentation', False),
    'use_coral': advanced_config.get('use_coral', False),
    'use_ensemble': advanced_config.get('use_ensemble', False),
    'use_contrastive': advanced_config.get('use_contrastive', False),
    # Hyperparameters for advanced techniques
    'ewc_lambda': ewc_lambda if advanced_config.get('use_ewc', False) else 0,
    'coral_lambda': coral_lambda if advanced_config.get('use_coral', False) else 0,
    'contrastive_lambda': contrastive_lambda if advanced_config.get('use_contrastive', False) else 0
}
torch.save(custom_metrics, f'{experiment_dir}/custom_metrics.pt')

# =============================================================================
# FINAL SUMMARY AND VISUALIZATION
# =============================================================================
print("\n" + "="*60)
print("TRAINING SUMMARY")
print("="*60)

print(f"Target participant: {args.target_participant}")
print(f"Base participants: {base_participants}")
print(f"\nPhase 1 (Base Training):")
print(f"  • Best {base_metric_name}: {base_best_metric:.4f}")
print(f"  • Test F1 on target participant: {base_model_test_f1:.4f}")
print(f"  • Training samples (train+val): {sum(base_train_info.values()):,}")

print(f"\nPhase 2 (Customization):")
print(f"  • Best {custom_metric_name}: {custom_best_metric:.4f}")
print(f"  • Test F1 on target participant: {custom_model_test_f1:.4f}")
print(f"  • Training samples: {sum(base_train_info.values()) + sum(target_train_info.values()):,}")

print(f"\nFinal Performance Improvement:")
print(f"  • Absolute improvement: {absolute_improvement:+.4f}")
print(f"  • Percentage improvement: {percentage_improvement:+.2f}%")

# Create final combined visualization with separate base and target validation curves
# Combine both phases into single arrays with proper indexing
combined_trainlossi = base_trainlossi + custom_trainlossi
combined_targetlossi = target_testlossi_phase1 + custom_target_testlossi  # Continuous target test
combined_trainf1i = base_trainf1i + custom_trainf1i
combined_targetf1i = target_testf1i_phase1 + custom_target_testf1i  # Continuous target test F1

# Create separate base and target validation curves that run through both phases
# Now using actual evaluation data from both phases
phase1_length = len(base_vallossi)
phase2_length = len(custom_target_vallossi)
total_length = phase1_length + phase2_length

# Base validation curves (actual evaluation in both phases)
combined_basevalidationlossi = base_vallossi + base_vallossi_phase2  # Actual base validation throughout
combined_basevalidationf1i = base_valf1i + base_valf1i_phase2  # Actual base validation F1 throughout

# Target validation curves (actual evaluation in both phases)
combined_targetvalidationlossi = target_vallossi_phase1 + custom_target_vallossi  # Actual target validation throughout
combined_targetvalidationf1i = target_valf1i_phase1 + custom_target_valf1i  # Actual target validation F1 throughout

# Calculate transition point (where phase 1 ends and phase 2 begins)
transition_epoch = len(base_trainf1i) - 1

# Define descriptive labels for final combined view with separate base/target training curves
final_labels = {
    'train': 'Combined Training (Legacy)',  # Legacy parameter
    'target': 'Target Test Log Loss (Continuous Evaluation)',
    'base_val': 'Base Validation Log Loss (Continuous Evaluation)',
    'target_val': 'Target Validation Log Loss (Continuous Evaluation)',
    'train_f1': 'Combined Training F1 (Legacy)',  # Legacy parameter
    'target_f1': 'Target Test F1 (Continuous Evaluation)',
    'base_val_f1': 'Base Validation F1 (Continuous Evaluation)', 
    'target_val_f1': 'Target Validation F1 (Continuous Evaluation)',
    # New separate training labels
    'base_train': 'Base Training Log Loss (Phase 1)',
    'target_train': 'Target Training Log Loss (Phase 2)',
    'base_train_f1': 'Base Training F1 (Phase 1)',
    'target_train_f1': 'Target Training F1 (Phase 2)'
}

# Create best model annotations
best_models = {
    'phase1': {
        'epoch': base_best_epoch,
        'metric_value': base_best_metric,
        'metric_name': base_metric_name
    },
    'phase2': {
        'epoch': custom_best_epoch,
        'metric_value': custom_best_metric,
        'metric_name': custom_metric_name
    }
}

plot_training_progress(
    trainlossi=combined_trainlossi,  # Legacy - keep for backward compatibility
    testlossi=None,  # Legacy parameter - not used since we have separate base/target validation
    targetlossi=combined_targetlossi,  # Continuous target test performance throughout both phases
    basevalidationlossi=combined_basevalidationlossi,  # Base validation: active Phase 1, plateau Phase 2
    targetvalidationlossi=combined_targetvalidationlossi,  # Target validation: pre-Phase 1, active Phase 2
    trainf1i=combined_trainf1i,  # Legacy - keep for backward compatibility
    testf1i=None,  # Legacy parameter - not used since we have separate base/target validation
    targetf1i=combined_targetf1i,  # Continuous target test performance throughout both phases
    basevalidationf1i=combined_basevalidationf1i,  # Base validation F1: active Phase 1, plateau Phase 2
    targetvalidationf1i=combined_targetvalidationf1i,  # Target validation F1: pre-Phase 1, active Phase 2
    # New separate training parameters (continuous evaluation)
    base_trainlossi=base_train_continuous_lossi,  # Base training loss (continuous evaluation)
    target_trainlossi=target_train_continuous_lossi,  # Target training loss (continuous evaluation)
    base_trainf1i=base_train_continuous_f1i,  # Base training F1 (continuous evaluation)
    target_trainf1i=target_train_continuous_f1i,  # Target training F1 (continuous evaluation)
    ma_window_size=config['visualization']['ma_window_size'],
    save_path=f'{experiment_dir}/final_combined_training_plot.{config["visualization"]["plot_format"]}',
    transition_epoch=transition_epoch,
    custom_labels=final_labels,
    best_models=best_models
)

print(f"\nVisualization saved: {experiment_dir}/combined_training_plot.{config['visualization']['plot_format']}")

# Save summary report
summary_report = f"""Two-Phase Customization Training Results
========================================

Target Participant: {args.target_participant}
Base Participants: {', '.join(base_participants)}

Dataset Information:
  • Base training samples (train+val): {sum(base_train_info.values()):,}
  • Base validation samples (test): {sum(base_val_info.values()):,}
  • Target training samples: {sum(target_train_info.values()):,}
  • Target validation samples: {sum(target_val_info.values()):,}
  • Target test samples: {sum(target_test_info.values()):,}

Phase 1 Results (Base Training):
  • Best Validation Metric: {base_best_metric:.4f}
  • Test F1 on Target Participant: {base_model_test_f1:.4f}
  • Epochs trained: {len(base_trainf1i)}
  • Early stopping patience: {base_patience}

Phase 2 Results (Customization):
  • Best Validation Metric: {custom_best_metric:.4f}
  • Test F1 on Target Participant: {custom_model_test_f1:.4f}
  • Epochs trained: {len(custom_trainf1i)}
  • Learning rate: {custom_lr}
  • Early stopping patience: {custom_patience}

Performance Improvement:
  • Absolute improvement: {absolute_improvement:+.4f}
  • Percentage improvement: {percentage_improvement:+.2f}%

Advanced Techniques Used:
{f"  • {', '.join(enabled_techniques)}" if enabled_techniques else "  • None (baseline approach)"}

Files Generated:
  • base_model.pt - Best model from Phase 1
  • customized_model.pt - Best model from Phase 2
  • base_metrics.pt - Phase 1 training history
  • custom_metrics.pt - Phase 2 training history
  • combined_training_plot.{config["visualization"]["plot_format"]} - Training visualization
"""

with open(f'{experiment_dir}/results_summary.txt', 'w') as f:
    f.write(summary_report)

print(f"\n✅ Two-phase training complete!")
print(f"   📁 Results saved in: {experiment_dir}")
print(f"   📈 Performance improvement: {absolute_improvement:+.4f} F1 score ({percentage_improvement:+.2f}%)")