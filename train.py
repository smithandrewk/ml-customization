"""
Two-phase leave-one-participant-out training script with base training + customization
"""
import os
import argparse
from time import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import numpy as np

from sklearn.metrics import f1_score

from utils import load_config, get_experiment_dir, get_next_experiment_dir, plot_training_progress, SmokingCNN, SimpleSmokingCNN, calculate_positive_ratio, init_final_layer_bias_for_imbalance
from dotenv import load_dotenv

load_dotenv()

# Parse command line arguments
parser = argparse.ArgumentParser(description='Two-phase customization training for smoking detection')
parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
parser.add_argument('--dataset_dir', type=str, required=True, help='Directory containing participant-specific train/test files')
parser.add_argument('--target_participant', type=str, required=True, help='Participant to customize for (hold out for phase 2)')
parser.add_argument('--experiment_suffix', type=str, help='Suffix for experiment name (default: custom_{target_participant})')
parser.add_argument('--model', type=str, default='full', choices=['full', 'simple'], help='Model architecture: full (RegNet-style) or simple (3-layer for testing)')
args = parser.parse_args()

# Load configuration
config = load_config(args.config)

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
            
            # For base participants doing validation, combine val + test for larger validation set
            if split == 'val' and is_base_participants:
                test_path = f'{dataset_dir}/{participant}_test.pt'
                if os.path.exists(test_path):
                    X_test, y_test = torch.load(test_path)
                    X = torch.cat([X, X_test], dim=0)
                    y = torch.cat([y, y_test], dim=0)
                    print(f"Combined {participant}_val.pt + {participant}_test.pt for base validation: {len(X):,} samples")
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

# Load base training data (N-1 participants)
base_train_datasets, base_train_info = load_participant_data(args.dataset_dir, base_participants, 'train')
base_val_datasets, base_val_info = load_participant_data(args.dataset_dir, base_participants, 'val', is_base_participants=True)

# Also load target participant test and validation data for evaluation throughout both phases
target_test_datasets, target_test_info = load_participant_data(args.dataset_dir, [args.target_participant], 'test')
target_val_datasets, target_val_info = load_participant_data(args.dataset_dir, [args.target_participant], 'val')

# Create dataloaders with clear names
base_trainloader = create_combined_dataloader(base_train_datasets, config['training']['batch_size'], shuffle=True)
base_valloader = create_combined_dataloader(base_val_datasets, config['training']['batch_size'], shuffle=False)
target_testloader = create_combined_dataloader(target_test_datasets, config['training']['batch_size'], shuffle=False)
target_valloader = create_combined_dataloader(target_val_datasets, config['training']['batch_size'], shuffle=False)

print(f"Base training samples: {sum(base_train_info.values()):,}")
print(f"Base validation samples: {sum(base_val_info.values()):,}")
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

# Phase 1 early stopping variables
base_n_epochs_without_improvement = 0
base_max_val_f1 = 0.0
base_patience = config['training'].get('base_patience', config['training'].get('patience', 10))

print(f"Starting Phase 1 training with patience={base_patience}")

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

    # Plot training progress every epoch for Phase 1 with 4-line format
    if epoch % 1 == 0:
        # Create current arrays for Phase 1 plotting (no Phase 2 data yet)
        current_basevalidationlossi = base_vallossi
        current_basevalidationf1i = base_valf1i
        current_targetvalidationlossi = target_vallossi_phase1
        current_targetvalidationf1i = target_valf1i_phase1
        
        # Define descriptive labels for Phase 1
        phase1_labels = {
            'train': 'Training Loss: Phase 1 (Base Training)',
            'target': 'Target Test Loss (Continuous Evaluation)',
            'base_val': 'Base Validation Loss (Continuous Evaluation)', 
            'target_val': 'Target Validation Loss (Continuous Evaluation)',
            'train_f1': 'Training F1: Phase 1 (Base Training)',
            'target_f1': 'Target Test F1 (Continuous Evaluation)',
            'base_val_f1': 'Base Validation F1 (Continuous Evaluation)',
            'target_val_f1': 'Target Validation F1 (Continuous Evaluation)'
        }
        
        plot_training_progress(
            trainlossi=base_trainlossi,
            testlossi=None,  # Legacy parameter - not used
            targetlossi=target_testlossi_phase1,  # Target test loss during base training
            basevalidationlossi=current_basevalidationlossi,  # Base validation loss
            targetvalidationlossi=current_targetvalidationlossi,  # Target validation loss
            trainf1i=base_trainf1i,
            testf1i=None,  # Legacy parameter - not used
            targetf1i=target_testf1i_phase1,  # Target test F1 during base training
            basevalidationf1i=current_basevalidationf1i,  # Base validation F1
            targetvalidationf1i=current_targetvalidationf1i,  # Target validation F1
            ma_window_size=config['visualization']['ma_window_size'],
            save_path=f'{experiment_dir}/training_progress.{config["visualization"]["plot_format"]}',
            custom_labels=phase1_labels
        )

    # Phase 1 early stopping (based on base validation F1)
    if base_valf1i[-1] > base_max_val_f1:
        base_max_val_f1 = base_valf1i[-1]
        torch.save(model.state_dict(), f'{experiment_dir}/base_model.pt')
        print(f"Phase 1 Epoch {epoch}: New best base model saved with val F1 score {base_max_val_f1:.4f}")
        base_n_epochs_without_improvement = 0
    else:
        print(f"Phase 1 Epoch {epoch}: No improvement ({base_valf1i[-1]:.4f} vs {base_max_val_f1:.4f})")
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

print(f"\nPhase 1 completed! Best base validation F1 score: {base_max_val_f1:.4f}")

# Save Phase 1 metrics
base_metrics = {
    'train_loss': base_trainlossi,
    'val_loss': base_vallossi,
    'target_test_loss': target_testlossi_phase1,
    'train_f1': base_trainf1i,
    'val_f1': base_valf1i,
    'target_test_f1': target_testf1i_phase1,
    'best_val_f1': base_max_val_f1,
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

# Load target participant data
target_train_datasets, target_train_info = load_participant_data(args.dataset_dir, [args.target_participant], 'train')
target_val_datasets, target_val_info = load_participant_data(args.dataset_dir, [args.target_participant], 'val')
# target_test_datasets already loaded in Phase 1

# Create combined training data (base + target)
combined_train_datasets = base_train_datasets + target_train_datasets
combined_trainloader = create_combined_dataloader(combined_train_datasets, config['training']['batch_size'], shuffle=True)

# Create validation dataloaders
target_valloader = create_combined_dataloader(target_val_datasets, config['training']['batch_size'], shuffle=False)
# Keep target_testloader from Phase 1 (already created)

print(f"Combined training samples: {sum(base_train_info.values()) + sum(target_train_info.values()):,}")
print(f"  - Base participants: {sum(base_train_info.values()):,}")
print(f"  - Target participant: {sum(target_train_info.values()):,}")
print(f"Target validation samples: {sum(target_val_info.values()):,}")
print(f"Target test samples: {sum(target_test_info.values()):,}")

# Calculate positive ratio from combined training data for Phase 2 bias initialization
combined_y_labels = []
for dataset in combined_train_datasets:
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
optimizer = optim.AdamW(
    model.parameters(), 
    lr=custom_lr,
    weight_decay=config['training']['weight_decay']
)

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
custom_max_val_f1 = 0.0
custom_patience = config['training'].get('custom_patience', config['training'].get('patience', 10) // 2)

print(f"Starting Phase 2 customization with LR={custom_lr}, patience={custom_patience}")

for epoch in range(config['training']['num_epochs']):
    epoch_start_time = time()
    model.train()

    train_preds = []
    train_labels = []
    train_losses = []

    for Xi, yi in combined_trainloader:
        Xi, yi = Xi.to(device), yi.to(device).float()
        optimizer.zero_grad()
        outputs = model(Xi).squeeze()
        loss = criterion(outputs, yi)
        loss.backward()
        optimizer.step()

        # Minimize GPU→CPU transfers - keep on GPU until end of epoch
        train_preds.append((outputs.sigmoid() > 0.5).float())
        train_labels.append(yi)
        train_losses.append(loss.item())

    # Single GPU→CPU transfer per epoch
    train_preds = torch.cat(train_preds).cpu()
    train_labels = torch.cat(train_labels).cpu()
    train_loss = np.mean(train_losses)
    custom_trainlossi.append(train_loss)
    custom_trainf1i.append(f1_score(train_labels, train_preds, average='macro', zero_division=0))

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
            'train': 'Training Loss: Phase 1 (Base) → Phase 2 (Target)',
            'target': 'Target Test Loss (Continuous Evaluation)',
            'base_val': 'Base Validation Loss (Continuous Evaluation)', 
            'target_val': 'Target Validation Loss (Continuous Evaluation)',
            'train_f1': 'Training F1: Phase 1 (Base) → Phase 2 (Target)',
            'target_f1': 'Target Test F1 (Continuous Evaluation)',
            'base_val_f1': 'Base Validation F1 (Continuous Evaluation)',
            'target_val_f1': 'Target Validation F1 (Continuous Evaluation)'
        }
        
        plot_training_progress(
            trainlossi=combined_trainlossi_current,
            testlossi=None,  # Legacy parameter - not used
            targetlossi=combined_targetlossi_current,  # Continuous target test
            basevalidationlossi=combined_basevalidationlossi_current,  # Continuous base validation
            targetvalidationlossi=combined_targetvalidationlossi_current,  # Continuous target validation
            trainf1i=combined_trainf1i_current,
            testf1i=None,  # Legacy parameter - not used
            targetf1i=combined_targetf1i_current,  # Continuous target test F1
            basevalidationf1i=combined_basevalidationf1i_current,  # Continuous base validation F1
            targetvalidationf1i=combined_targetvalidationf1i_current,  # Continuous target validation F1
            ma_window_size=config['visualization']['ma_window_size'],
            save_path=f'{experiment_dir}/training_progress.{config["visualization"]["plot_format"]}',
            transition_epoch=current_transition_epoch,
            custom_labels=phase2_labels
        )

    # Phase 2 early stopping (based on target validation performance)
    if custom_target_valf1i[-1] > custom_max_val_f1:
        custom_max_val_f1 = custom_target_valf1i[-1]
        torch.save(model.state_dict(), f'{experiment_dir}/customized_model.pt')
        print(f"Phase 2 Epoch {epoch}: New best customized model saved with target val F1 score {custom_max_val_f1:.4f}")
        custom_n_epochs_without_improvement = 0
    else:
        print(f"Phase 2 Epoch {epoch}: No improvement ({custom_target_valf1i[-1]:.4f} vs {custom_max_val_f1:.4f})")
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

print(f"\nPhase 2 completed! Best target validation F1 score: {custom_max_val_f1:.4f}")

# =============================================================================
# FINAL MODEL EVALUATION
# =============================================================================
print("\n" + "="*60)
print("FINAL MODEL EVALUATION")
print("="*60)

# Evaluate Phase 1 best model on target test set
print("Evaluating Phase 1 best model (base training) on target test set...")
model.load_state_dict(torch.load(f'{experiment_dir}/base_model.pt'))
model.eval()

base_model_test_preds = []
base_model_test_labels = []
base_model_test_losses = []

with torch.no_grad():
    for Xi, yi in target_testloader:
        Xi, yi = Xi.to(device), yi.to(device).float()
        outputs = model(Xi).squeeze()
        loss = criterion(outputs, yi)
        
        base_model_test_preds.append((outputs.sigmoid() > 0.5).float())
        base_model_test_labels.append(yi)
        base_model_test_losses.append(loss.item())

base_model_test_preds = torch.cat(base_model_test_preds).cpu()
base_model_test_labels = torch.cat(base_model_test_labels).cpu()
base_model_test_f1 = f1_score(base_model_test_labels, base_model_test_preds, average='macro', zero_division=0)

print(f"Phase 1 best model test F1: {base_model_test_f1:.4f}")

# Evaluate Phase 2 best model on target test set
print("Evaluating Phase 2 best model (customized) on target test set...")
model.load_state_dict(torch.load(f'{experiment_dir}/customized_model.pt'))
model.eval()

custom_model_test_preds = []
custom_model_test_labels = []
custom_model_test_losses = []

with torch.no_grad():
    for Xi, yi in target_testloader:
        Xi, yi = Xi.to(device), yi.to(device).float()
        outputs = model(Xi).squeeze()
        loss = criterion(outputs, yi)
        
        custom_model_test_preds.append((outputs.sigmoid() > 0.5).float())
        custom_model_test_labels.append(yi)
        custom_model_test_losses.append(loss.item())

custom_model_test_preds = torch.cat(custom_model_test_preds).cpu()
custom_model_test_labels = torch.cat(custom_model_test_labels).cpu()
custom_model_test_f1 = f1_score(custom_model_test_labels, custom_model_test_preds, average='macro', zero_division=0)

print(f"Phase 2 best model test F1: {custom_model_test_f1:.4f}")

# Calculate improvements
absolute_improvement = custom_model_test_f1 - base_model_test_f1
percentage_improvement = (absolute_improvement / base_model_test_f1 * 100) if base_model_test_f1 > 0 else 0.0

print(f"\nPerformance Improvement:")
print(f"  • Absolute improvement: {absolute_improvement:+.4f}")
print(f"  • Percentage improvement: {percentage_improvement:+.2f}%")

# Save Phase 2 metrics
custom_metrics = {
    'train_loss': custom_trainlossi,
    'target_val_loss': custom_target_vallossi,
    'target_test_loss': custom_target_testlossi,
    'train_f1': custom_trainf1i,
    'target_val_f1': custom_target_valf1i,
    'target_test_f1': custom_target_testf1i,
    'best_target_val_f1': custom_max_val_f1,
    'base_model_test_f1': base_model_test_f1,
    'custom_model_test_f1': custom_model_test_f1,
    'absolute_improvement': absolute_improvement,
    'percentage_improvement': percentage_improvement,
    'target_participant': args.target_participant,
    'combined_train_samples': sum(base_train_info.values()) + sum(target_train_info.values()),
    'target_val_samples': sum(target_val_info.values()),
    'target_test_samples': sum(target_test_info.values()),
    'custom_learning_rate': custom_lr
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
print(f"  • Best validation F1 on base participants: {base_max_val_f1:.4f}")
print(f"  • Test F1 on target participant: {base_model_test_f1:.4f}")
print(f"  • Training samples: {sum(base_train_info.values()):,}")

print(f"\nPhase 2 (Customization):")
print(f"  • Best validation F1 on target participant: {custom_max_val_f1:.4f}")
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

# Define descriptive labels for final combined view with 4 separate curves
final_labels = {
    'train': 'Training Loss: Phase 1 (Base) → Phase 2 (Target)',
    'target': 'Target Test Loss (Continuous Evaluation)',
    'base_val': 'Base Validation Loss (Continuous Evaluation)',
    'target_val': 'Target Validation Loss (Continuous Evaluation)',
    'train_f1': 'Training F1: Phase 1 (Base) → Phase 2 (Target)',
    'target_f1': 'Target Test F1 (Continuous Evaluation)',
    'base_val_f1': 'Base Validation F1 (Continuous Evaluation)', 
    'target_val_f1': 'Target Validation F1 (Continuous Evaluation)'
}

plot_training_progress(
    trainlossi=combined_trainlossi,
    testlossi=None,  # Legacy parameter - not used since we have separate base/target validation
    targetlossi=combined_targetlossi,  # Continuous target test performance throughout both phases
    basevalidationlossi=combined_basevalidationlossi,  # Base validation: active Phase 1, plateau Phase 2
    targetvalidationlossi=combined_targetvalidationlossi,  # Target validation: pre-Phase 1, active Phase 2
    trainf1i=combined_trainf1i,
    testf1i=None,  # Legacy parameter - not used since we have separate base/target validation
    targetf1i=combined_targetf1i,  # Continuous target test performance throughout both phases
    basevalidationf1i=combined_basevalidationf1i,  # Base validation F1: active Phase 1, plateau Phase 2
    targetvalidationf1i=combined_targetvalidationf1i,  # Target validation F1: pre-Phase 1, active Phase 2
    ma_window_size=config['visualization']['ma_window_size'],
    save_path=f'{experiment_dir}/final_combined_training_plot.{config["visualization"]["plot_format"]}',
    transition_epoch=transition_epoch,
    custom_labels=final_labels
)

print(f"\nVisualization saved: {experiment_dir}/combined_training_plot.{config['visualization']['plot_format']}")

# Save summary report
summary_report = f"""Two-Phase Customization Training Results
========================================

Target Participant: {args.target_participant}
Base Participants: {', '.join(base_participants)}

Dataset Information:
  • Base training samples: {sum(base_train_info.values()):,}
  • Base validation samples: {sum(base_val_info.values()):,}
  • Target training samples: {sum(target_train_info.values()):,}
  • Target validation samples: {sum(target_val_info.values()):,}
  • Target test samples: {sum(target_test_info.values()):,}

Phase 1 Results (Base Training):
  • Best Validation F1 Score: {base_max_val_f1:.4f}
  • Test F1 on Target Participant: {base_model_test_f1:.4f}
  • Epochs trained: {len(base_trainf1i)}
  • Early stopping patience: {base_patience}

Phase 2 Results (Customization):
  • Best Validation F1 Score: {custom_max_val_f1:.4f}
  • Test F1 on Target Participant: {custom_model_test_f1:.4f}
  • Epochs trained: {len(custom_trainf1i)}
  • Learning rate: {custom_lr}
  • Early stopping patience: {custom_patience}

Performance Improvement:
  • Absolute improvement: {absolute_improvement:+.4f}
  • Percentage improvement: {percentage_improvement:+.2f}%

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