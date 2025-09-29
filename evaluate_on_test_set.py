#!/usr/bin/env python3
"""
Evaluate trained models on held-out test sets.

This script loads saved model checkpoints and computes performance on test sets
that were never used during training or early stopping. This provides unbiased
estimates of model performance for manuscript reporting.

Usage:
    python evaluate_on_test_set.py --experiment <experiment_name>
    python evaluate_on_test_set.py --all  # Process all experiments
"""

import os
import json
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import argparse
import glob

# Import model architectures and utilities
from lib.utils import SimpleSmokingCNN, MediumSmokingCNN, SmokingCNN
from lib.utils_simple import compute_loss_and_f1


def evaluate_fold(fold_dir, verbose=True):
    """
    Evaluate a single fold on its test set.

    Args:
        fold_dir: Path to fold directory (e.g., 'experiments/alpha/fold1_asfik')
        verbose: Whether to print progress messages

    Returns:
        dict: Updated metrics with test set results
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluating: {fold_dir}")
        print(f"{'='*60}")

    # Load hyperparameters
    hyperparams_file = os.path.join(fold_dir, 'hyperparameters.json')
    if not os.path.exists(hyperparams_file):
        print(f"  ❌ No hyperparameters.json found in {fold_dir}")
        return None

    with open(hyperparams_file, 'r') as f:
        hyperparams = json.load(f)

    # Load existing metrics
    metrics_file = os.path.join(fold_dir, 'metrics.json')
    if not os.path.exists(metrics_file):
        print(f"  ❌ No metrics.json found in {fold_dir}")
        return None

    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    # Check if test metrics already exist
    if 'base_test_f1' in metrics and 'test_f1' in metrics:
        if verbose:
            print(f"  ⚠️  Test metrics already exist. Overwriting...")

    # Extract parameters
    target_participant = hyperparams['target_participant']
    data_path = hyperparams['data_path']
    model_type = hyperparams['model_type']
    window_size = hyperparams.get('window_size', 3000)
    device = hyperparams.get('device', 'cuda:0')
    batch_size = hyperparams.get('batch_size', 128)

    if verbose:
        print(f"  Target participant: {target_participant}")
        print(f"  Model type: {model_type}")
        print(f"  Device: {device}")

    # Load test dataset
    test_data_file = f'{data_path}/{target_participant}_test.pt'
    if not os.path.exists(test_data_file):
        print(f"  ❌ Test data not found: {test_data_file}")
        return None

    X_test, y_test = torch.load(test_data_file)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if verbose:
        print(f"  Test set size: {len(test_dataset)} samples")

    # Instantiate model
    if model_type == 'simple':
        model = SimpleSmokingCNN(window_size=window_size, num_features=6)
    elif model_type == 'medium':
        model = MediumSmokingCNN(window_size=window_size, num_features=6)
    elif model_type == 'full':
        model = SmokingCNN(window_size=window_size, num_features=6)
    elif model_type == 'test':
        # TestModel architecture from train.py
        class ConvLayerNorm(nn.Module):
            def __init__(self, out_channels):
                super(ConvLayerNorm, self).__init__()
                self.ln = nn.LayerNorm(out_channels, elementwise_affine=False)

            def forward(self, x):
                x = x.permute(0, 2, 1)
                x = self.ln(x)
                x = x.permute(0, 2, 1)
                return x

        class Block(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, pool_size=2, pool=True):
                super(Block, self).__init__()
                self.pool_flag = pool
                self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
                self.ln = ConvLayerNorm(out_channels)
                if self.pool_flag:
                    self.pool = nn.MaxPool1d(pool_size)

            def forward(self, x):
                x = self.conv(x)
                x = self.ln(x)
                x = torch.relu(x)
                if self.pool_flag:
                    x = self.pool(x)
                return x

        class TestModel(nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()
                self.blocks = []
                self.blocks.append(Block(6, 8))
                for _ in range(5):
                    self.blocks.append(Block(8, 8))
                    self.blocks.append(Block(8, 8, pool=False))
                self.blocks.append(Block(8, 16, pool=False))
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
        print(f"  ❌ Unknown model type: {model_type}")
        return None

    criterion = nn.BCEWithLogitsLoss()
    model.to(device)

    # Evaluate base model on test set
    base_model_file = os.path.join(fold_dir, 'best_base_model.pt')
    if os.path.exists(base_model_file):
        model.load_state_dict(torch.load(base_model_file, map_location=device))
        model.eval()
        base_test_loss, base_test_f1 = compute_loss_and_f1(model, test_loader, criterion, device=device)
        metrics['base_test_loss'] = base_test_loss
        metrics['base_test_f1'] = base_test_f1
        if verbose:
            print(f"  ✓ Base model test F1: {base_test_f1:.4f}")
    else:
        print(f"  ⚠️  No best_base_model.pt found - skipping base model evaluation")
        metrics['base_test_loss'] = None
        metrics['base_test_f1'] = None

    # Evaluate target model on test set
    target_model_file = os.path.join(fold_dir, 'best_target_model.pt')
    if os.path.exists(target_model_file):
        model.load_state_dict(torch.load(target_model_file, map_location=device))
        model.eval()
        test_loss, test_f1 = compute_loss_and_f1(model, test_loader, criterion, device=device)
        metrics['test_loss'] = test_loss
        metrics['test_f1'] = test_f1
        if verbose:
            print(f"  ✓ Personalized model test F1: {test_f1:.4f}")
            if base_test_f1 is not None:
                improvement = test_f1 - base_test_f1
                rel_improvement = (improvement / base_test_f1) * 100 if base_test_f1 > 0 else 0
                print(f"  ✓ Improvement: {improvement:+.4f} ({rel_improvement:+.1f}%)")
    else:
        print(f"  ⚠️  No best_target_model.pt found - skipping target model evaluation")
        metrics['test_loss'] = None
        metrics['test_f1'] = None

    # Save updated metrics
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)

    if verbose:
        print(f"  ✓ Updated metrics saved to {metrics_file}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate models on held-out test sets')
    parser.add_argument('--experiment', type=str, help='Specific experiment name to evaluate')
    parser.add_argument('--all', action='store_true', help='Evaluate all experiments')
    parser.add_argument('--device', type=str, default=None, help='Override device (e.g., cuda:0, cpu)')
    args = parser.parse_args()

    if not args.experiment and not args.all:
        print("Error: Must specify either --experiment <name> or --all")
        parser.print_help()
        return

    # Find experiments to process
    experiments_dir = 'experiments'
    if not os.path.exists(experiments_dir):
        print(f"Error: {experiments_dir} directory not found")
        return

    if args.all:
        experiments = [d for d in os.listdir(experiments_dir)
                      if os.path.isdir(os.path.join(experiments_dir, d))]
        print(f"Found {len(experiments)} experiments to process")
    else:
        experiments = [args.experiment]
        exp_path = os.path.join(experiments_dir, args.experiment)
        if not os.path.exists(exp_path):
            print(f"Error: Experiment {args.experiment} not found in {experiments_dir}")
            return

    # Process each experiment
    total_folds = 0
    successful_folds = 0

    for experiment in experiments:
        experiment_dir = os.path.join(experiments_dir, experiment)
        print(f"\n{'#'*60}")
        print(f"# Processing experiment: {experiment}")
        print(f"{'#'*60}")

        # Find all fold directories
        fold_dirs = glob.glob(os.path.join(experiment_dir, 'fold*'))

        if not fold_dirs:
            print(f"  No fold directories found in {experiment_dir}")
            continue

        print(f"Found {len(fold_dirs)} folds in {experiment}")

        for fold_dir in sorted(fold_dirs):
            total_folds += 1

            # Override device if specified
            if args.device:
                hyperparams_file = os.path.join(fold_dir, 'hyperparameters.json')
                if os.path.exists(hyperparams_file):
                    with open(hyperparams_file, 'r') as f:
                        hyperparams = json.load(f)
                    hyperparams['device'] = args.device
                    with open(hyperparams_file, 'w') as f:
                        json.dump(hyperparams, f, indent=4)

            result = evaluate_fold(fold_dir, verbose=True)
            if result is not None:
                successful_folds += 1

    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total folds processed: {total_folds}")
    print(f"Successful evaluations: {successful_folds}")
    print(f"Failed evaluations: {total_folds - successful_folds}")
    print(f"\n✓ Test set evaluation complete!")
    print(f"\nNext steps:")
    print(f"  1. Run: python manuscript/generate_results.py <experiment_name>")
    print(f"  2. Recompile manuscript to see updated results")


if __name__ == "__main__":
    main()