import os
from sympy import hyper
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

from lib.utils_simple import *
import argparse
import json
from time import time

argparser = argparse.ArgumentParser()
argparser.add_argument('--fold', type=int, required=True, help='Fold index for leave-one-participant-out cross-validation')
argparser.add_argument('--device', type=int, required=True, default=0, help='GPU device index')
argparser.add_argument('--batch_size', type=int, required=True, default=64, help='batch size')
argparser.add_argument('--model', type=str, default='medium', choices=['simple', 'medium', 'full', 'test'],
                      help='Model architecture: simple (SimpleSmokingCNN), medium (MediumSmokingCNN), full (SmokingCNN)')
argparser.add_argument('--use_augmentation', action='store_true', help='Enable data augmentation')
argparser.add_argument('--jitter_std', type=float, default=0.005, help='Standard deviation for jitter noise')
argparser.add_argument('--magnitude_range', type=float, nargs=2, default=[0.98, 1.02], help='Range for magnitude scaling')
argparser.add_argument('--aug_prob', type=float, default=0.3, help='Probability of applying augmentation')
argparser.add_argument('--prefix', type=str, default='alpha', help='Experiment prefix/directory name')
argparser.add_argument('--early_stopping_patience', type=int, default=40, help='Early stopping patience for base phase')
argparser.add_argument('--early_stopping_patience_target', type=int, default=40, help='Early stopping patience for target phase')
argparser.add_argument('--mode', type=str, default='full_fine_tuning', choices=['full_fine_tuning', 'last_layer_only', 'generic', 'target_only'], help='Mode')
argparser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
argparser.add_argument('--target_data_pct', type=float, default=1.0, help='Percentage of target training data to use (0.0-1.0)')
args = argparser.parse_args()

hyperparameters = {
    'fold': args.fold,
    'device':f'cuda:{args.device}',
    'lr': args.lr,
    'batch_size': args.batch_size,
    'early_stopping_patience': args.early_stopping_patience,
    'early_stopping_patience_target': args.early_stopping_patience_target,
    'window_size': 3000,
    'participants': ['tonmoy','asfik','ejaz'],
    # 'participants': ['tonmoy','alsaad','anam','asfik','ejaz','iftakhar','unk1','dennis'],
    'experiment_prefix': args.prefix,
    'target_participant': None,  # to be set later
    'data_path': 'data/001_test',
    'model_type': args.model,
    'use_augmentation': args.use_augmentation,
    'jitter_std': args.jitter_std,
    'magnitude_range': args.magnitude_range,
    'aug_prob': args.aug_prob,
    'mode': args.mode,
    'target_data_pct': args.target_data_pct
    }

fold = hyperparameters['fold']
device = hyperparameters['device']
batch_size = hyperparameters['batch_size']
window_size = hyperparameters['window_size']
experiment_prefix = hyperparameters['experiment_prefix']
data_path = hyperparameters['data_path']
participants = hyperparameters['participants']
target_participant = participants[fold]
hyperparameters['target_participant'] = target_participant

new_exp_dir = f'experiments/{experiment_prefix}/fold{fold}_{target_participant}'
os.makedirs(new_exp_dir, exist_ok=False)
# new_exp_dir = create_and_get_new_exp_dir(prefix=experiment_prefix)

if hyperparameters['mode'] == 'generic':
    print("Generic mode: using ALL participants including target in training data.")
    # Keep all participants - don't remove target
elif hyperparameters['mode'] == 'target_only':
    print(f"Target-only mode: training only on {target_participant} data.")
    participants = []  # Empty - no other participants
else:
    print(f"Leave-one-participant-out mode: using {target_participant} as target participant.")
    participants.remove(target_participant)

if participants:  # Not empty (generic or personalization modes)
    base_train_dataset = ConcatDataset([TensorDataset(*torch.load(f'{data_path}/{p}_{s}.pt')) for p in participants for s in ['train', 'val']])
    base_val_dataset = ConcatDataset([TensorDataset(*torch.load(f'{data_path}/{p}_test.pt')) for p in participants])
else:  # target_only mode
    # Use target data as "base" data (we'll only train on this)
    base_train_dataset = TensorDataset(*torch.load(f'{data_path}/{target_participant}_train.pt'))
    base_val_dataset = TensorDataset(*torch.load(f'{data_path}/{target_participant}_val.pt'))

target_train_dataset = TensorDataset(*torch.load(f'{data_path}/{target_participant}_train.pt'))
target_val_dataset = TensorDataset(*torch.load(f'{data_path}/{target_participant}_val.pt'))
target_test_dataset = TensorDataset(*torch.load(f'{data_path}/{target_participant}_test.pt'))

# Subsample target training data if specified
target_data_pct = hyperparameters['target_data_pct']
if target_data_pct < 1.0:
    original_size = len(target_train_dataset)
    subset_size = int(original_size * target_data_pct)

    # Create random indices for subsampling
    import random
    random.seed(42)  # For reproducibility
    indices = random.sample(range(original_size), subset_size)

    # Create subset dataset
    from torch.utils.data import Subset
    target_train_dataset = Subset(target_train_dataset, indices)

    # In target_only mode, also subsample base_train_dataset (which is the same data)
    if hyperparameters['mode'] == 'target_only':
        base_train_dataset = Subset(base_train_dataset, indices)

    print(f'Subsampled target training data: {original_size} -> {subset_size} samples ({target_data_pct*100:.1f}%)')

trainloader = DataLoader(base_train_dataset, batch_size=batch_size, shuffle=True)
base_valloader = DataLoader(base_val_dataset, batch_size=batch_size)
target_trainloader = DataLoader(target_train_dataset, batch_size=batch_size)
target_valloader = DataLoader(target_val_dataset, batch_size=batch_size)

print(f'Base train dataset size: {len(base_train_dataset)}')
print(f'Base val dataset size: {len(base_val_dataset)}')
print(f'Target train dataset size: {len(target_train_dataset)}')
print(f'Target val dataset size: {len(target_val_dataset)}')
print(f'Target test dataset size: {len(target_test_dataset)}')

from lib.utils import SimpleSmokingCNN, MediumSmokingCNN, SmokingCNN

model_type = hyperparameters['model_type']
if model_type == 'simple':
    model = SimpleSmokingCNN(window_size=window_size, num_features=6)
elif model_type == 'medium':
    model = MediumSmokingCNN(window_size=window_size, num_features=6)
elif model_type == 'full':
    model = SmokingCNN(window_size=window_size, num_features=6)
elif model_type == 'test':
    class ConvLayerNorm(nn.Module):
        # might actually be instance norm haha jun19
        
        def __init__(self, out_channels) -> None:
            super(ConvLayerNorm,self).__init__()
            self.ln = nn.LayerNorm(out_channels, elementwise_affine=False)

        def forward(self,x):
            x = x.permute(0, 2, 1)
            x = self.ln(x)
            x = x.permute(0, 2, 1)
            return x

    class Block(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, pool_size=2, pool=True) -> None:
            super(Block,self).__init__()
            self.pool = pool
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
            self.ln = ConvLayerNorm(out_channels)
            if self.pool:
                self.pool = nn.MaxPool1d(pool_size)

        def forward(self,x):
            x = self.conv(x)
            x = self.ln(x)
            x = torch.relu(x)
            if self.pool:
                x = self.pool(x)
            return x
    class TestModel(nn.Module):
        def __init__(self):
            super(TestModel, self).__init__()
            self.blocks = []
            self.blocks.append(Block(6,8))
            for _ in range(5):
                self.blocks.append(Block(8,8))
                self.blocks.append(Block(8,8,pool=False))

            self.blocks.append(Block(8,16,pool=False))

            # for _ in range(5):
            #     self.blocks.append(Block(16,16))
            #     self.blocks.append(Block(16,16,pool=False))
                
            self.blocks = nn.ModuleList(self.blocks)
            self.gap = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(16, 1)
        
        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            
            x = self.gap(x).squeeze(-1)
            x = self.fc(x)
            return x
    model = TestModel()
else:
    raise ValueError(f"Invalid model type: {model_type}. Choose from 'simple', 'medium', 'full'")
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparameters['lr'])
print(f'Using {model_type} model: {model.__class__.__name__}')

augmenter = None
if hyperparameters['use_augmentation']:
    from lib.utils import TimeSeriesAugmenter
    augmenter = TimeSeriesAugmenter({
        'jitter_noise_std': hyperparameters['jitter_std'],
        'magnitude_scale_range': hyperparameters['magnitude_range'],
        'augmentation_probability': hyperparameters['aug_prob']
    })
    print(f'Data augmentation enabled: jitter_std={hyperparameters["jitter_std"]}, mag_range={hyperparameters["magnitude_range"]}, prob={hyperparameters["aug_prob"]}')

print(f'Total model parameters: {sum(p.numel() for p in model.parameters())}')

metrics = {
    'transition_epoch': None,
    'best_base_val_loss': None,
    'best_base_val_loss_epoch': None,
    'best_target_val_loss_epoch': None,
    'best_target_val_loss_epoch': None,
    'best_base_val_f1': None,
    'best_base_val_f1_epoch': None,
    'best_target_val_f1': None,
    'best_target_val_f1_epoch': None,
}

lossi = {
    'base train loss': [],
    'base train f1': [],
    'base val loss': [],
    'base val f1': [],
    'target train loss': [],
    'target train f1': [],
    'target val loss': [],
    'target val f1': [],
}

model.to(device)
epoch = 0
best_val_loss = float('inf')
patience_counter = 0
phase = 'base'

while True:
    start_time = time()
    model.train()
    train_loss, train_f1 = optimize_model_compute_loss_and_f1(model, trainloader, optimizer, criterion, device=device, augmenter=augmenter)

    lossi['base train loss'].append(train_loss)
    lossi['base train f1'].append(train_f1)

    loss,f1 = compute_loss_and_f1(model, base_valloader, criterion, device=device)
    lossi['base val loss'].append(loss)
    lossi['base val f1'].append(f1)

    loss,f1 = compute_loss_and_f1(model, target_trainloader, criterion, device=device)
    lossi['target train loss'].append(loss)
    lossi['target train f1'].append(f1)

    loss,f1 = compute_loss_and_f1(model, target_valloader, criterion, device=device)
    lossi['target val loss'].append(loss)
    lossi['target val f1'].append(f1)

    # For generic mode, use target val loss for early stopping; otherwise use phase-specific val loss
    val_loss_key = 'target val loss' if hyperparameters['mode'] == 'generic' else f'{phase} val loss'
    val_f1_key = 'target val f1' if hyperparameters['mode'] == 'generic' else f'{phase} val f1'

    if lossi[val_loss_key][-1] < best_val_loss:
        best_val_loss = lossi[val_loss_key][-1]
        metrics[f'best_{phase}_val_loss'] = best_val_loss
        metrics[f'best_{phase}_val_loss_epoch'] = epoch
        torch.save(model.state_dict(), f'{new_exp_dir}/best_{phase}_model.pt')
        patience_counter = 0
    else:
        patience_counter += 1

    if lossi[val_f1_key][-1] > (metrics[f'best_{phase}_val_f1'] or 0):
        metrics[f'best_{phase}_val_f1'] = lossi[val_f1_key][-1]
        metrics[f'best_{phase}_val_f1_epoch'] = epoch

    if patience_counter >= hyperparameters['early_stopping_patience' if phase == 'base' else 'early_stopping_patience_target']:
        if phase == 'base' and hyperparameters['mode'] not in ['generic', 'target_only']:
            torch.save(model.state_dict(), f'{new_exp_dir}/last_{phase}_model.pt')
            phase = 'target'
            trainloader = DataLoader(ConcatDataset([base_train_dataset, target_train_dataset]), batch_size=batch_size, shuffle=True)
            best_val_loss = float('inf')
            patience_counter = 0
            
            metrics['transition_epoch'] = epoch
            metrics['best_target_val_loss_from_best_base_model'] = lossi['target val loss'][metrics['best_base_val_loss_epoch']]
            metrics['best_target_val_f1_from_best_base_model'] = lossi['target val f1'][metrics['best_base_val_loss_epoch']]

            if hyperparameters['mode'] == 'last_layer_only':
                for param in model.parameters():
                    param.requires_grad = False
                if hasattr(model, 'fc'):
                    for param in model.fc.parameters():
                        param.requires_grad = True
                elif hasattr(model, 'classifier'):
                    for param in model.classifier.parameters():
                        param.requires_grad = True
                else:
                    raise ValueError("Model does not have 'fc' or 'classifier' attribute for last layer fine-tuning.")
                print("Fine-tuning only the last layer of the model.")
                print(f'Trainable model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
        else:
            print("Early stopping triggered.")
            torch.save(model.state_dict(), f'{new_exp_dir}/last_{phase}_model.pt')

            with open(f'{new_exp_dir}/metrics.json', 'w') as f:
                json.dump(metrics, f, indent=4)
            with open(f'{new_exp_dir}/losses.json', 'w') as f:
                json.dump(lossi, f, indent=4)
            with open(f'{new_exp_dir}/hyperparameters.json', 'w') as f:
                json.dump(hyperparameters, f, indent=4)
            break

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
        plt.axhline(y=lossi['target val f1'][metrics['best_base_val_f1_epoch']], color='g', linestyle='--', label='Best Base Val Loss', alpha=0.5)

    if metrics['best_target_val_f1_epoch'] is not None and metrics['best_target_val_f1'] is not None:
        plt.axhline(y=metrics['best_target_val_f1'], color='g', linestyle='--', label='Best target f1 Loss', alpha=0.8)
        plt.axvline(x=metrics['best_target_val_f1_epoch'], color='g', linestyle='--', alpha=0.4)

    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.legend()
    plt.title(f'Patience Counter: {patience_counter}')
    plt.savefig(f'{new_exp_dir}/f1.jpg', bbox_inches='tight')
    plt.close()

    epoch += 1
    print(f'Epoch {epoch}, Phase: {phase}, Time Elapsed: {time() - start_time:.2f}s, Patience Counter: {patience_counter}, Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}, Val Loss: {lossi[val_loss_key][-1]:.4f}, Val F1: {lossi[val_f1_key][-1]:.4f}')