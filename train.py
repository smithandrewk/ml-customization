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

from utils import load_config, get_experiment_dir, get_next_experiment_dir, plot_training_progress, SmokingCNN, calculate_positive_ratio, init_final_layer_bias_for_imbalance
from dotenv import load_dotenv

load_dotenv()

# Parse command line arguments
parser = argparse.ArgumentParser(description='Two-phase customization training for smoking detection')
parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
parser.add_argument('--dataset_dir', type=str, required=True, help='Directory containing participant-specific train/test files')
parser.add_argument('--target_participant', type=str, required=True, help='Participant to customize for (hold out for phase 2)')
parser.add_argument('--experiment_suffix', type=str, help='Suffix for experiment name (default: custom_{target_participant})')
args = parser.parse_args()

# Load configuration
config = load_config(args.config)

# Extract dataset name from dataset_dir for new experiment naming
# e.g., "data/001_tonmoy_60s" -> "tonmoy_60s"
dataset_name = os.path.basename(args.dataset_dir).split('_', 1)[1] if '_' in os.path.basename(args.dataset_dir) else os.path.basename(args.dataset_dir)
experiment_dir = get_next_experiment_dir(f"{dataset_name}_custom_{args.target_participant}", "custom")

def load_participant_data(dataset_dir, participants, split='train'):
    """Load and return separate datasets for each participant."""
    datasets = []
    participant_info = {}
    
    for participant in participants:
        file_path = f'{dataset_dir}/{participant}_{split}.pt'
        if os.path.exists(file_path):
            X, y = torch.load(file_path)
            datasets.append(TensorDataset(X, y))
            participant_info[participant] = len(X)
            print(f"Loaded {participant}_{split}.pt: {len(X):,} samples")
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
base_test_datasets, base_test_info = load_participant_data(args.dataset_dir, base_participants, 'test')

# Also load target participant test data for evaluation throughout both phases
target_test_datasets_phase1, target_test_info_phase1 = load_participant_data(args.dataset_dir, [args.target_participant], 'test')

base_trainloader = create_combined_dataloader(base_train_datasets, config['training']['batch_size'], shuffle=True)
base_testloader = create_combined_dataloader(base_test_datasets, config['training']['batch_size'], shuffle=False)
target_testloader_phase1 = create_combined_dataloader(target_test_datasets_phase1, config['training']['batch_size'], shuffle=False)

print(f"Base training samples: {sum(base_train_info.values()):,}")
print(f"Base test samples: {sum(base_test_info.values()):,}")

# Initialize model
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
base_testlossi = []
base_trainf1i = []
base_testf1i = []

# Target test metrics for Phase 1 (to track target performance during base training)
target_testlossi_phase1 = []
target_testf1i_phase1 = []

# Phase 1 early stopping variables
base_n_epochs_without_improvement = 0
base_max_test_f1 = 0.0
base_patience = config['training'].get('base_patience', config['training']['patience'])

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

    # Base test evaluation
    model.eval()
    test_preds = []
    test_labels = []
    test_losses = []

    with torch.no_grad():
        for Xi, yi in base_testloader:
            Xi, yi = Xi.to(device), yi.to(device).float()
            outputs = model(Xi).squeeze()
            loss = criterion(outputs, yi)

            test_preds.append((outputs.sigmoid() > 0.5).float())
            test_labels.append(yi)
            test_losses.append(loss.item())

    test_preds = torch.cat(test_preds).cpu()
    test_labels = torch.cat(test_labels).cpu()
    test_loss = np.mean(test_losses)
    base_testlossi.append(test_loss)
    base_testf1i.append(f1_score(test_labels, test_preds, average='macro', zero_division=0))

    # Also evaluate on target test set during Phase 1 (to see how base training affects target performance)
    target_test_preds = []
    target_test_labels = []
    target_test_losses = []

    with torch.no_grad():
        for Xi, yi in target_testloader_phase1:
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

    # Phase 1 early stopping
    if base_testf1i[-1] > base_max_test_f1:
        base_max_test_f1 = base_testf1i[-1]
        torch.save(model.state_dict(), f'{experiment_dir}/base_model.pt')
        print(f"Phase 1 Epoch {epoch}: New best base model saved with F1 score {base_max_test_f1:.4f}")
        base_n_epochs_without_improvement = 0
    else:
        print(f"Phase 1 Epoch {epoch}: No improvement ({base_testf1i[-1]:.4f} vs {base_max_test_f1:.4f})")
        base_n_epochs_without_improvement += 1

    if base_n_epochs_without_improvement >= base_patience:
        print(f"Phase 1 early stopping triggered after {base_n_epochs_without_improvement} epochs without improvement.")
        break
        
    # Plot training progress every epoch for Phase 1 (including target test performance)
    if epoch % 1 == 0:
        plot_training_progress(
            trainlossi=base_trainlossi,
            testlossi=base_testlossi,
            targetlossi=target_testlossi_phase1,  # Target test loss during base training
            trainf1i=base_trainf1i,
            testf1i=base_testf1i,
            targetf1i=target_testf1i_phase1,  # Target test F1 during base training
            ma_window_size=config['visualization']['ma_window_size'],
            save_path=f'{experiment_dir}/phase1_training_plot.{config["visualization"]["plot_format"]}',
        )

    epoch_time = time() - epoch_start_time
    print(f"Phase 1 Epoch {epoch} completed in {epoch_time:.2f}s. "
          f"Train Loss: {train_loss:.4f}, Base Test Loss: {test_loss:.4f}, Target Test Loss: {target_test_loss:.4f}, "
          f"Train F1: {base_trainf1i[-1]:.4f}, Base Test F1: {base_testf1i[-1]:.4f}, Target Test F1: {target_testf1i_phase1[-1]:.4f}. "
          f"Epochs without improvement: {base_n_epochs_without_improvement}")

print(f"\nPhase 1 completed! Best base F1 score: {base_max_test_f1:.4f}")

# Save Phase 1 metrics
base_metrics = {
    'train_loss': base_trainlossi,
    'test_loss': base_testlossi,
    'train_f1': base_trainf1i,
    'test_f1': base_testf1i,
    'best_f1': base_max_test_f1,
    'base_participants': base_participants,
    'base_train_samples': sum(base_train_info.values()),
    'base_test_samples': sum(base_test_info.values())
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
target_test_datasets, target_test_info = load_participant_data(args.dataset_dir, [args.target_participant], 'test')

# Create combined training data (base + target)
combined_train_datasets = base_train_datasets + target_train_datasets
combined_trainloader = create_combined_dataloader(combined_train_datasets, config['training']['batch_size'], shuffle=True)

# Test only on target participant
target_testloader = create_combined_dataloader(target_test_datasets, config['training']['batch_size'], shuffle=False)

print(f"Combined training samples: {sum(base_train_info.values()) + sum(target_train_info.values()):,}")
print(f"  - Base participants: {sum(base_train_info.values()):,}")
print(f"  - Target participant: {sum(target_train_info.values()):,}")
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

# Re-initialize final layer bias for Phase 2 with updated class distribution
init_final_layer_bias_for_imbalance(model, combined_positive_ratio)

# Create new optimizer for customization phase
custom_lr = config['training'].get('custom_learning_rate', config['training']['learning_rate'] * 0.1)
optimizer = optim.AdamW(
    model.parameters(), 
    lr=custom_lr,
    weight_decay=config['training']['weight_decay']
)

# Phase 2 training metrics
custom_trainlossi = []
custom_target_testlossi = []
custom_trainf1i = []
custom_target_testf1i = []

# Phase 2 early stopping variables
custom_n_epochs_without_improvement = 0
custom_max_test_f1 = 0.0
custom_patience = config['training'].get('custom_patience', config['training']['patience'] // 2)

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

    # Target test evaluation
    model.eval()
    test_preds = []
    test_labels = []
    test_losses = []

    with torch.no_grad():
        for Xi, yi in target_testloader:
            Xi, yi = Xi.to(device), yi.to(device).float()
            outputs = model(Xi).squeeze()
            loss = criterion(outputs, yi)

            test_preds.append((outputs.sigmoid() > 0.5).float())
            test_labels.append(yi)
            test_losses.append(loss.item())

    test_preds = torch.cat(test_preds).cpu()
    test_labels = torch.cat(test_labels).cpu()
    test_loss = np.mean(test_losses)
    custom_target_testlossi.append(test_loss)
    custom_target_testf1i.append(f1_score(test_labels, test_preds, average='macro', zero_division=0))

    # Phase 2 early stopping (based on target test performance)
    if custom_target_testf1i[-1] > custom_max_test_f1:
        custom_max_test_f1 = custom_target_testf1i[-1]
        torch.save(model.state_dict(), f'{experiment_dir}/customized_model.pt')
        print(f"Phase 2 Epoch {epoch}: New best customized model saved with target F1 score {custom_max_test_f1:.4f}")
        custom_n_epochs_without_improvement = 0
    else:
        print(f"Phase 2 Epoch {epoch}: No improvement ({custom_target_testf1i[-1]:.4f} vs {custom_max_test_f1:.4f})")
        custom_n_epochs_without_improvement += 1

    if custom_n_epochs_without_improvement >= custom_patience:
        print(f"Phase 2 early stopping triggered after {custom_n_epochs_without_improvement} epochs without improvement.")
        break
        
    # Plot training progress every epoch for Phase 2 (with combined view and continuous target curve)
    if epoch % 1 == 0:
        # Create combined arrays for Phase 2 plotting
        combined_phase2_trainlossi = base_trainlossi + custom_trainlossi
        # Keep base test performance for orange line throughout both phases
        combined_phase2_testlossi = base_testlossi + [base_testlossi[-1]] * len(custom_trainlossi)  # Extend base test performance
        combined_phase2_targetlossi = target_testlossi_phase1 + custom_target_testlossi  # Continuous target curve
        combined_phase2_trainf1i = base_trainf1i + custom_trainf1i
        # Keep base test F1 for orange line throughout both phases  
        combined_phase2_testf1i = base_testf1i + [base_testf1i[-1]] * len(custom_trainf1i)  # Extend base test F1
        combined_phase2_targetf1i = target_testf1i_phase1 + custom_target_testf1i  # Continuous target curve
        phase2_transition_epoch = len(base_trainf1i) - 1
        
        plot_training_progress(
            trainlossi=combined_phase2_trainlossi,
            testlossi=combined_phase2_testlossi,  # Continuous base test performance 
            targetlossi=combined_phase2_targetlossi,  # Continuous target performance
            trainf1i=combined_phase2_trainf1i,
            testf1i=combined_phase2_testf1i,  # Continuous base test performance
            targetf1i=combined_phase2_targetf1i,  # Continuous target performance
            ma_window_size=config['visualization']['ma_window_size'],
            save_path=f'{experiment_dir}/combined_training_plot.{config["visualization"]["plot_format"]}',
            transition_epoch=phase2_transition_epoch
        )

    epoch_time = time() - epoch_start_time
    print(f"Phase 2 Epoch {epoch} completed in {epoch_time:.2f}s. "
          f"Train Loss: {train_loss:.4f}, Target Test Loss: {test_loss:.4f}, "
          f"Train F1: {custom_trainf1i[-1]:.4f}, Target Test F1: {custom_target_testf1i[-1]:.4f}. "
          f"Epochs without improvement: {custom_n_epochs_without_improvement}")

print(f"\nPhase 2 completed! Best target F1 score: {custom_max_test_f1:.4f}")

# Save Phase 2 metrics
custom_metrics = {
    'train_loss': custom_trainlossi,
    'target_test_loss': custom_target_testlossi,
    'train_f1': custom_trainf1i,
    'target_test_f1': custom_target_testf1i,
    'best_target_f1': custom_max_test_f1,
    'target_participant': args.target_participant,
    'combined_train_samples': sum(base_train_info.values()) + sum(target_train_info.values()),
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
print(f"  • Best F1 on base participants: {base_max_test_f1:.4f}")
print(f"  • Training samples: {sum(base_train_info.values()):,}")

print(f"\nPhase 2 (Customization):")
print(f"  • Best F1 on target participant: {custom_max_test_f1:.4f}")
print(f"  • Improvement: {custom_max_test_f1 - base_max_test_f1:+.4f}")
print(f"  • Training samples: {sum(base_train_info.values()) + sum(target_train_info.values()):,}")

# Create final combined visualization with continuous curves for both base and target
# Combine both phases into single arrays with proper indexing
combined_trainlossi = base_trainlossi + custom_trainlossi
# Keep base test performance for orange line throughout both phases (flat after Phase 1)
combined_testlossi = base_testlossi + [base_testlossi[-1]] * len(custom_trainlossi)  # Continuous base test
combined_targetlossi = target_testlossi_phase1 + custom_target_testlossi  # Continuous target performance
combined_trainf1i = base_trainf1i + custom_trainf1i
# Keep base test F1 for orange line throughout both phases (flat after Phase 1)
combined_testf1i = base_testf1i + [base_testf1i[-1]] * len(custom_trainf1i)  # Continuous base test
combined_targetf1i = target_testf1i_phase1 + custom_target_testf1i  # Continuous target performance

# Calculate transition point (where phase 1 ends and phase 2 begins)
transition_epoch = len(base_trainf1i) - 1

plot_training_progress(
    trainlossi=combined_trainlossi,
    testlossi=combined_testlossi,  # Shows continuous base test performance (flat in Phase 2)
    targetlossi=combined_targetlossi,  # Shows continuous target test performance throughout both phases
    trainf1i=combined_trainf1i,
    testf1i=combined_testf1i,  # Shows continuous base test performance (flat in Phase 2)
    targetf1i=combined_targetf1i,  # Shows continuous target test performance throughout both phases
    ma_window_size=config['visualization']['ma_window_size'],
    save_path=f'{experiment_dir}/final_combined_training_plot.{config["visualization"]["plot_format"]}',
    transition_epoch=transition_epoch
)

print(f"\nVisualization saved: {experiment_dir}/combined_training_plot.{config['visualization']['plot_format']}")

# Save summary report
summary_report = f"""Two-Phase Customization Training Results
========================================

Target Participant: {args.target_participant}
Base Participants: {', '.join(base_participants)}

Dataset Information:
  • Base training samples: {sum(base_train_info.values()):,}
  • Base test samples: {sum(base_test_info.values()):,}
  • Target training samples: {sum(target_train_info.values()):,}
  • Target test samples: {sum(target_test_info.values()):,}

Phase 1 Results (Base Training):
  • Best F1 Score: {base_max_test_f1:.4f}
  • Epochs trained: {len(base_trainf1i)}
  • Early stopping patience: {base_patience}

Phase 2 Results (Customization):
  • Best F1 Score: {custom_max_test_f1:.4f}
  • Improvement over base: {custom_max_test_f1 - base_max_test_f1:+.4f}
  • Epochs trained: {len(custom_trainf1i)}
  • Learning rate: {custom_lr}
  • Early stopping patience: {custom_patience}

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
print(f"   📈 Performance improvement: {custom_max_test_f1 - base_max_test_f1:+.4f} F1 score")