#!/usr/bin/env python3
"""
Comprehensive Training Dynamics Analysis Tool

Analyzes training health, overfitting, model capacity, and transfer learning quality
for ML experiments. Goes beyond final performance metrics to understand training dynamics.

Authors: Based on techniques from Google, OpenAI, Meta, and academic literature
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from scipy import signal, stats
from scipy.optimize import curve_fit
import argparse
import warnings
warnings.filterwarnings('ignore')

class TrainingAnalyzer:
    """Comprehensive training dynamics analyzer for ML experiments."""

    def __init__(self, experiment_path: str, run_id: str):
        """Initialize analyzer with paths to training data."""
        self.experiment_path = experiment_path
        self.run_id = run_id
        self.run_dir = f"{experiment_path}/{run_id}"

        # Load training data
        self.hyperparameters = self._load_json('hyperparameters.json')
        self.metrics = self._load_json('metrics.json')
        self.losses = self._load_json('losses.json')

        # Extract key training curves
        self.transition_epoch = self.metrics.get('transition_epoch', 0)
        self.extract_training_curves()

        # Computed metrics storage
        self.training_health = {}

    def _load_json(self, filename: str) -> Dict:
        """Load JSON file from run directory."""
        filepath = f"{self.run_dir}/{filename}"
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Required file not found: {filepath}")

        with open(filepath, 'r') as f:
            return json.load(f)

    def extract_training_curves(self):
        """Extract and organize training curves from losses.json."""
        # Base model phase (training on source data)
        self.base_train_loss = np.array(self.losses.get('base train loss', []))
        self.base_val_loss = np.array(self.losses.get('base val loss', []))

        # Target model phase (training on target data)
        self.target_train_loss = np.array(self.losses.get('target train loss', []))
        self.target_val_loss = np.array(self.losses.get('target val loss', []))

        # Verify that all curves have the same length
        curves = [self.base_train_loss, self.base_val_loss, self.target_train_loss, self.target_val_loss]
        curve_lengths = [len(curve) for curve in curves if len(curve) > 0]

        if len(set(curve_lengths)) > 1:
            print(f"Warning: Curves have different lengths: {curve_lengths}")

        # Use the length of base training as reference (all should be same)
        self.epochs_per_phase = len(self.base_train_loss) if len(self.base_train_loss) > 0 else 0
        self.total_epochs = self.epochs_per_phase * 2  # Base + Target phases

        # Create epoch indices for each phase
        self.base_epochs = np.arange(self.epochs_per_phase)
        self.target_epochs = np.arange(self.epochs_per_phase, self.total_epochs)
        self.all_epochs = np.arange(self.total_epochs)

        # In transfer learning, all curves have same length and start from epoch 0
        # All four curves are evaluated every epoch, transition only changes what we train on
        if len(self.base_train_loss) > 0:
            # Use base training loss as reference for overfitting analysis
            # (since this is what we're actually optimizing before transition)
            self.full_train_loss = np.array(self.base_train_loss)
            self.full_val_loss = np.array(self.base_val_loss)
            self.epochs = np.arange(len(self.base_train_loss))
        elif len(self.target_train_loss) > 0:
            # Fallback to target if base not available
            self.full_train_loss = np.array(self.target_train_loss)
            self.full_val_loss = np.array(self.target_val_loss)
            self.epochs = np.arange(len(self.target_train_loss))
        else:
            self.full_train_loss = np.array([])
            self.full_val_loss = np.array([])
            self.epochs = np.array([])

    def compute_overfitting_metrics(self) -> Dict[str, float]:
        """Compute comprehensive overfitting and generalization metrics."""
        metrics = {}

        # 1. Train-Validation Gap Analysis
        train_val_gap = self.full_val_loss - self.full_train_loss

        # Overall gap statistics
        metrics['mean_train_val_gap'] = np.mean(train_val_gap)
        metrics['max_train_val_gap'] = np.max(train_val_gap)
        metrics['final_train_val_gap'] = train_val_gap[-1] if len(train_val_gap) > 0 else 0

        # Gap progression (is it getting worse?)
        if len(train_val_gap) > 10:
            early_gap = np.mean(train_val_gap[:len(train_val_gap)//3])
            late_gap = np.mean(train_val_gap[-len(train_val_gap)//3:])
            metrics['gap_progression'] = late_gap - early_gap  # Positive = getting worse
        else:
            metrics['gap_progression'] = 0

        # 2. Validation Loss Trajectory Analysis
        if len(self.full_val_loss) > 5:
            # Check if validation loss is increasing in later epochs
            mid_point = len(self.full_val_loss) // 2
            early_val = np.mean(self.full_val_loss[:mid_point])
            late_val = np.mean(self.full_val_loss[mid_point:])
            metrics['val_loss_increase'] = late_val - early_val

            # Minimum validation loss position (early = good, late = overfitting risk)
            min_val_epoch = np.argmin(self.full_val_loss)
            metrics['min_val_position'] = min_val_epoch / len(self.full_val_loss)  # 0-1 scale
        else:
            metrics['val_loss_increase'] = 0
            metrics['min_val_position'] = 0.5

        # 3. Overfitting Score (0-100, higher = more overfitting)
        # Combines multiple factors
        gap_score = min(100, max(0, metrics['mean_train_val_gap'] * 50))  # Scale gap
        progression_score = min(50, max(0, metrics['gap_progression'] * 100))  # Scale progression
        position_score = max(0, (metrics['min_val_position'] - 0.7) * 100)  # Penalty for late minimum

        metrics['overfitting_score'] = gap_score + progression_score + position_score

        return metrics

    def compute_learning_efficiency_metrics(self) -> Dict[str, float]:
        """Compute learning efficiency and convergence quality metrics."""
        metrics = {}

        # 1. Learning Speed Analysis
        if len(self.full_train_loss) > 10:
            # Early learning rate (first 25% of training)
            early_epochs = len(self.full_train_loss) // 4
            early_loss_reduction = self.full_train_loss[0] - self.full_train_loss[early_epochs]
            metrics['early_learning_speed'] = early_loss_reduction / early_epochs

            # Overall learning rate
            total_loss_reduction = self.full_train_loss[0] - self.full_train_loss[-1]
            metrics['overall_learning_speed'] = total_loss_reduction / len(self.full_train_loss)

            # Learning efficiency (how much of the total reduction happened early)
            if total_loss_reduction > 0:
                metrics['learning_efficiency'] = early_loss_reduction / total_loss_reduction
            else:
                metrics['learning_efficiency'] = 0
        else:
            metrics['early_learning_speed'] = 0
            metrics['overall_learning_speed'] = 0
            metrics['learning_efficiency'] = 0

        # 2. Loss Curve Smoothness (stability indicator)
        if len(self.full_train_loss) > 5:
            # Compute loss derivatives (change per epoch)
            train_derivatives = np.diff(self.full_train_loss)
            val_derivatives = np.diff(self.full_val_loss)

            # Smoothness = inverse of variance in derivatives
            train_smoothness = 1 / (1 + np.var(train_derivatives))
            val_smoothness = 1 / (1 + np.var(val_derivatives))

            metrics['train_curve_smoothness'] = train_smoothness
            metrics['val_curve_smoothness'] = val_smoothness
            metrics['overall_smoothness'] = (train_smoothness + val_smoothness) / 2
        else:
            metrics['train_curve_smoothness'] = 1.0
            metrics['val_curve_smoothness'] = 1.0
            metrics['overall_smoothness'] = 1.0

        # 3. Convergence Quality
        if len(self.full_train_loss) > 20:
            # Look at final 25% of training for convergence
            final_portion = len(self.full_train_loss) // 4
            final_train_losses = self.full_train_loss[-final_portion:]
            final_val_losses = self.full_val_loss[-final_portion:]

            # Convergence = low variance in final losses
            train_convergence = 1 / (1 + np.var(final_train_losses))
            val_convergence = 1 / (1 + np.var(final_val_losses))

            metrics['train_convergence'] = train_convergence
            metrics['val_convergence'] = val_convergence
            metrics['convergence_quality'] = (train_convergence + val_convergence) / 2
        else:
            metrics['train_convergence'] = 1.0
            metrics['val_convergence'] = 1.0
            metrics['convergence_quality'] = 1.0

        return metrics

    def compute_transfer_learning_metrics(self) -> Dict[str, float]:
        """Compute transfer learning specific metrics for source->target transfer."""
        metrics = {}

        # Check if we have both base and target phases
        if len(self.base_val_loss) == 0 or len(self.target_val_loss) == 0:
            return {
                'adaptation_speed': 0,
                'knowledge_retention': 1.0,
                'transfer_efficiency': 0,
                'catastrophic_forgetting': 0,
                'transfer_gap': 0,
                'target_improvement_rate': 0
            }

        # 1. Transfer Gap - Performance drop when switching from base to target data
        base_final_performance = self.base_val_loss[-1]  # End of base training
        target_initial_performance = self.target_val_loss[0]  # Start of target training

        # Positive gap means performance got worse (expected for new domain)
        transfer_gap = target_initial_performance - base_final_performance
        metrics['transfer_gap'] = transfer_gap

        # 2. Adaptation Speed - How quickly target performance improves
        if len(self.target_val_loss) > 5:
            # Compare early vs total improvement on target
            early_cutoff = min(len(self.target_val_loss) // 4, 10)  # First 25% or 10 epochs

            target_start = self.target_val_loss[0]
            target_early = self.target_val_loss[early_cutoff]
            target_final = self.target_val_loss[-1]

            total_target_improvement = target_start - target_final
            early_target_improvement = target_start - target_early

            if total_target_improvement > 0:
                metrics['adaptation_speed'] = early_target_improvement / total_target_improvement
            else:
                metrics['adaptation_speed'] = 0
        else:
            metrics['adaptation_speed'] = 0

        # 3. Knowledge Retention/Transfer Quality
        # Compare how well the model maintained base performance level on target domain
        if transfer_gap > 0:
            # How much of the gap was recovered?
            target_improvement = self.target_val_loss[0] - self.target_val_loss[-1]
            if transfer_gap > 0:
                gap_recovery = min(1.0, target_improvement / transfer_gap)
                metrics['knowledge_retention'] = gap_recovery
            else:
                metrics['knowledge_retention'] = 1.0
        else:
            # No gap or negative gap (target started better) = excellent transfer
            metrics['knowledge_retention'] = 1.0

        # 4. Transfer Efficiency - Performance improvement per epoch on target
        if len(self.target_val_loss) > 1:
            target_improvement = self.target_val_loss[0] - self.target_val_loss[-1]
            target_epochs = len(self.target_val_loss)
            metrics['transfer_efficiency'] = max(0, target_improvement / target_epochs)
        else:
            metrics['transfer_efficiency'] = 0

        # 5. Target Improvement Rate - How much target performance improved
        target_total_improvement = self.target_val_loss[0] - self.target_val_loss[-1]
        metrics['target_improvement_rate'] = max(0, target_total_improvement)

        # 6. Catastrophic Forgetting Risk (updated interpretation)
        # Large transfer gap indicates potential forgetting of base knowledge
        # Normalize by base performance to make it relative
        if base_final_performance > 0:
            normalized_forgetting = transfer_gap / base_final_performance
            metrics['catastrophic_forgetting'] = max(0, normalized_forgetting)
        else:
            metrics['catastrophic_forgetting'] = 0

        # 7. Overall Transfer Success Score
        # Combines multiple factors: small gap, fast adaptation, good retention
        gap_score = max(0, 1 - (transfer_gap / (base_final_performance + 1e-8)))  # Penalty for large gap
        adaptation_score = metrics['adaptation_speed']
        retention_score = metrics['knowledge_retention']

        metrics['transfer_success_score'] = (gap_score + adaptation_score + retention_score) / 3

        return metrics

    def compute_capacity_metrics(self) -> Dict[str, float]:
        """Compute model capacity and utilization metrics."""
        metrics = {}

        # 1. Training Speed Analysis - Very fast training may indicate underfitting
        if len(self.full_train_loss) > 10:
            # Measure how quickly loss drops initially
            initial_loss = self.full_train_loss[0]
            loss_10_epochs = self.full_train_loss[min(9, len(self.full_train_loss)-1)]

            initial_drop_rate = (initial_loss - loss_10_epochs) / min(10, len(self.full_train_loss))

            # Very fast drops might indicate underfitting (too easy)
            # Very slow drops might indicate underfitting (insufficient capacity)
            metrics['initial_drop_rate'] = initial_drop_rate

            # Check if training plateaued early (underfitting sign)
            mid_epoch = len(self.full_train_loss) // 2
            early_avg = np.mean(self.full_train_loss[:mid_epoch])
            late_avg = np.mean(self.full_train_loss[mid_epoch:])

            improvement_second_half = early_avg - late_avg
            metrics['late_stage_improvement'] = improvement_second_half

        else:
            metrics['initial_drop_rate'] = 0
            metrics['late_stage_improvement'] = 0

        # 2. Loss Level Analysis
        # Very high final losses suggest underfitting
        # Very low training losses with high validation losses suggest overfitting
        if len(self.full_train_loss) > 0:
            metrics['final_train_loss'] = self.full_train_loss[-1]
            metrics['final_val_loss'] = self.full_val_loss[-1]

            # Capacity utilization score (heuristic)
            # Good models have moderate final losses with small train-val gaps
            final_gap = self.full_val_loss[-1] - self.full_train_loss[-1]

            # Normalize by initial loss to make it dataset-independent
            initial_loss = self.full_train_loss[0]
            normalized_final_train = self.full_train_loss[-1] / initial_loss
            normalized_gap = final_gap / initial_loss

            # Ideal: final loss around 0.1-0.5 of initial, gap < 0.1 of initial
            train_score = 1.0 - abs(normalized_final_train - 0.3)  # Peak at 0.3
            gap_score = max(0, 1.0 - normalized_gap * 10)  # Penalty for large gaps

            metrics['capacity_utilization_score'] = (train_score + gap_score) / 2
        else:
            metrics['final_train_loss'] = 0
            metrics['final_val_loss'] = 0
            metrics['capacity_utilization_score'] = 0.5

        return metrics

    def compute_training_health_score(self) -> float:
        """Compute overall training health score (0-100)."""
        # Combine all metrics into a single health score

        # Weights for different aspects
        weights = {
            'overfitting': 0.3,
            'learning_efficiency': 0.25,
            'convergence': 0.2,
            'transfer_quality': 0.15,
            'capacity_utilization': 0.1
        }

        # Overfitting (lower is better)
        overfitting_penalty = min(1.0, self.training_health.get('overfitting_score', 0) / 100)
        overfitting_contribution = (1 - overfitting_penalty) * weights['overfitting']

        # Learning efficiency (higher is better)
        efficiency_score = self.training_health.get('learning_efficiency', 0)
        smoothness_score = self.training_health.get('overall_smoothness', 0)
        efficiency_contribution = ((efficiency_score + smoothness_score) / 2) * weights['learning_efficiency']

        # Convergence (higher is better)
        convergence_score = self.training_health.get('convergence_quality', 0)
        convergence_contribution = convergence_score * weights['convergence']

        # Transfer quality (higher is better, lower forgetting is better)
        adaptation_score = self.training_health.get('adaptation_speed', 0)
        retention_score = self.training_health.get('knowledge_retention', 1.0)
        forgetting_penalty = min(1.0, self.training_health.get('catastrophic_forgetting', 0))
        transfer_score = (adaptation_score + retention_score - forgetting_penalty) / 2
        transfer_contribution = max(0, transfer_score) * weights['transfer_quality']

        # Capacity utilization (higher is better)
        capacity_score = self.training_health.get('capacity_utilization_score', 0.5)
        capacity_contribution = capacity_score * weights['capacity_utilization']

        # Final health score
        total_score = (overfitting_contribution +
                      efficiency_contribution +
                      convergence_contribution +
                      transfer_contribution +
                      capacity_contribution)

        return min(100, max(0, total_score * 100))

    def analyze_training(self) -> Dict[str, Any]:
        """Run complete training analysis and return all metrics."""
        print(f"Analyzing training run: {self.experiment_path}/{self.run_id}")

        # Compute all metric categories
        overfitting_metrics = self.compute_overfitting_metrics()
        efficiency_metrics = self.compute_learning_efficiency_metrics()
        transfer_metrics = self.compute_transfer_learning_metrics()
        capacity_metrics = self.compute_capacity_metrics()

        # Store all metrics
        self.training_health.update(overfitting_metrics)
        self.training_health.update(efficiency_metrics)
        self.training_health.update(transfer_metrics)
        self.training_health.update(capacity_metrics)

        # Compute overall health score
        health_score = self.compute_training_health_score()
        self.training_health['overall_health_score'] = health_score

        # Add metadata
        self.training_health['experiment_path'] = self.experiment_path
        self.training_health['run_id'] = self.run_id
        self.training_health['total_epochs'] = self.total_epochs
        self.training_health['transition_epoch'] = self.transition_epoch

        return self.training_health

    def generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []

        # Overfitting recommendations
        if self.training_health.get('overfitting_score', 0) > 50:
            recommendations.append("🚨 HIGH OVERFITTING DETECTED")
            if self.training_health.get('mean_train_val_gap', 0) > 0.1:
                recommendations.append("• Consider reducing model complexity or adding regularization")
            if self.training_health.get('gap_progression', 0) > 0.05:
                recommendations.append("• Implement early stopping or reduce learning rate")

        # Learning efficiency recommendations
        if self.training_health.get('learning_efficiency', 0) < 0.3:
            recommendations.append("⚡ SLOW INITIAL LEARNING")
            recommendations.append("• Consider increasing learning rate or improving initialization")

        if self.training_health.get('overall_smoothness', 0) < 0.5:
            recommendations.append("📈 UNSTABLE TRAINING")
            recommendations.append("• Consider reducing learning rate or increasing batch size")

        # Convergence recommendations
        if self.training_health.get('convergence_quality', 0) < 0.5:
            recommendations.append("🎯 POOR CONVERGENCE")
            recommendations.append("• Training may benefit from more epochs or learning rate scheduling")

        # Transfer learning recommendations
        if self.training_health.get('catastrophic_forgetting', 0) > 0.5:
            recommendations.append("🧠 CATASTROPHIC FORGETTING RISK")
            recommendations.append("• Consider lower learning rates during fine-tuning")

        if self.training_health.get('adaptation_speed', 0) < 0.3:
            recommendations.append("🐌 SLOW ADAPTATION")
            recommendations.append("• Consider higher learning rate for target task or longer training")

        # Capacity recommendations
        if self.training_health.get('capacity_utilization_score', 0) < 0.4:
            recommendations.append("🏗️ SUBOPTIMAL MODEL CAPACITY")
            if self.training_health.get('final_train_loss', 1) > 0.5:
                recommendations.append("• Model may be underfitted - consider increasing capacity")
            else:
                recommendations.append("• Model may be overfitted - consider reducing capacity")

        # Overall health recommendations
        health_score = self.training_health.get('overall_health_score', 0)
        if health_score > 80:
            recommendations.append("✅ EXCELLENT TRAINING HEALTH")
        elif health_score > 60:
            recommendations.append("👍 GOOD TRAINING HEALTH")
        elif health_score > 40:
            recommendations.append("⚠️ MODERATE TRAINING ISSUES")
        else:
            recommendations.append("❌ SIGNIFICANT TRAINING PROBLEMS")

        return recommendations

    def create_training_dashboard(self, save_path: Optional[str] = None):
        """Create comprehensive training dynamics visualization dashboard."""
        # Set up the plotting style
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 9,
            'axes.labelsize': 10,
            'axes.titlesize': 11,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 8,
            'figure.titlesize': 14
        })

        # Create 3x3 dashboard
        fig, axes = plt.subplots(3, 3, figsize=(16, 12))
        fig.suptitle(f'Training Dynamics Analysis: {self.experiment_path}/{self.run_id}',
                     fontsize=16, fontweight='bold')

        # Color scheme
        colors = {
            'train': '#1f77b4',
            'val': '#d62728',
            'transition': '#2ca02c',
            'highlight': '#ff7f0e',
            'neutral': '#7f7f7f'
        }

        # Panel 1: All Loss Curves from Epoch 0
        ax1 = axes[0, 0]

        # All curves start from epoch 0 and have same length
        epochs = np.arange(len(self.base_train_loss))

        # Plot all 4 curves
        if len(self.base_train_loss) > 0:
            ax1.plot(epochs, self.base_train_loss, color=colors['train'],
                    label='Base Train', linewidth=2)
            ax1.plot(epochs, self.base_val_loss, color=colors['train'],
                    label='Base Val', linewidth=2, linestyle='--')

        if len(self.target_train_loss) > 0:
            ax1.plot(epochs, self.target_train_loss, color=colors['val'],
                    label='Target Train', linewidth=2)
            ax1.plot(epochs, self.target_val_loss, color=colors['val'],
                    label='Target Val', linewidth=2, linestyle='--')

        # Mark the transition epoch (when training switches to target data)
        if hasattr(self, 'transition_epoch') and self.transition_epoch > 0:
            ax1.axvline(x=self.transition_epoch, color=colors['transition'],
                       linestyle=':', alpha=0.8, linewidth=3, label='Switch Training Data')

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training & Evaluation Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Panel 2: Train-Val Gap Over Time
        ax2 = axes[0, 1]
        train_val_gap = self.full_val_loss - self.full_train_loss
        ax2.plot(self.epochs, train_val_gap, color=colors['highlight'],
                linewidth=2, label='Val - Train Gap')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)

        if self.transition_epoch > 0:
            ax2.axvline(x=self.transition_epoch, color=colors['transition'],
                       linestyle='--', alpha=0.8)

        # Highlight overfitting regions
        overfitting_threshold = np.percentile(train_val_gap, 75)
        ax2.fill_between(self.epochs, train_val_gap, overfitting_threshold,
                        where=(train_val_gap > overfitting_threshold),
                        alpha=0.3, color='red', label='High Overfitting Risk')

        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Validation - Training Loss')
        ax2.set_title('Overfitting Analysis')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Panel 3: Learning Rate Analysis (Loss Derivatives)
        ax3 = axes[0, 2]
        if len(self.full_train_loss) > 1:
            train_derivatives = -np.diff(self.full_train_loss)  # Negative for improvement
            val_derivatives = -np.diff(self.full_val_loss)

            # Smooth derivatives for better visualization
            if len(train_derivatives) > 5:
                window = min(5, len(train_derivatives) // 5)
                train_smooth = np.convolve(train_derivatives, np.ones(window)/window, mode='valid')
                val_smooth = np.convolve(val_derivatives, np.ones(window)/window, mode='valid')
                smooth_epochs = self.epochs[window//2:len(train_smooth)+window//2]

                ax3.plot(smooth_epochs, train_smooth, color=colors['train'],
                        label='Train Improvement Rate', linewidth=2)
                ax3.plot(smooth_epochs, val_smooth, color=colors['val'],
                        label='Val Improvement Rate', linewidth=2)

            if self.transition_epoch > 0:
                ax3.axvline(x=self.transition_epoch, color=colors['transition'],
                           linestyle='--', alpha=0.8)

        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss Improvement per Epoch')
        ax3.set_title('Learning Speed Analysis')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Panel 4: Training Health Metrics Radar-style
        ax4 = axes[1, 0]
        metrics_names = ['Overfitting\n(Lower Better)', 'Learning\nEfficiency',
                        'Convergence\nQuality', 'Training\nSmoothness', 'Transfer\nQuality']

        # Normalize metrics to 0-1 scale for visualization
        overfitting_score = 1 - min(1.0, self.training_health.get('overfitting_score', 0) / 100)
        efficiency_score = self.training_health.get('learning_efficiency', 0)
        convergence_score = self.training_health.get('convergence_quality', 0)
        smoothness_score = self.training_health.get('overall_smoothness', 0)

        # Transfer quality (average of relevant metrics)
        transfer_metrics = [
            self.training_health.get('adaptation_speed', 0),
            self.training_health.get('knowledge_retention', 1.0),
            1 - min(1.0, self.training_health.get('catastrophic_forgetting', 0))
        ]
        transfer_score = np.mean(transfer_metrics)

        scores = [overfitting_score, efficiency_score, convergence_score,
                 smoothness_score, transfer_score]

        bars = ax4.bar(range(len(metrics_names)), scores,
                      color=[colors['val'], colors['train'], colors['highlight'],
                            colors['neutral'], colors['transition']], alpha=0.7)

        # Color code bars based on quality
        for i, (bar, score) in enumerate(zip(bars, scores)):
            if score > 0.8:
                bar.set_color('green')
            elif score > 0.6:
                bar.set_color('orange')
            else:
                bar.set_color('red')
            bar.set_alpha(0.7)

        ax4.set_xticks(range(len(metrics_names)))
        ax4.set_xticklabels(metrics_names, rotation=45, ha='right')
        ax4.set_ylabel('Score (0-1)')
        ax4.set_title('Training Health Metrics')
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3)

        # Panel 5: Target Domain Performance Analysis
        ax5 = axes[1, 1]
        if len(self.target_val_loss) > 0:
            # All curves start from epoch 0 and have same length
            epochs = np.arange(len(self.target_val_loss))

            # Plot target domain performance throughout training (evaluated from epoch 0)
            ax5.plot(epochs, self.target_val_loss, color=colors['val'],
                    linewidth=3, label='Target Val Loss')

            # Mark the transition epoch (when training switches to target data)
            if hasattr(self, 'transition_epoch') and self.transition_epoch > 0:
                ax5.axvline(x=self.transition_epoch, color=colors['transition'],
                           linestyle='--', linewidth=2, label='Switch to Training on Target')

                # Show performance at transition point and final
                transition_perf = self.target_val_loss[self.transition_epoch]
                final_perf = self.target_val_loss[-1]
                improvement = transition_perf - final_perf

                # Add annotations
                ax5.annotate(f'At transition: {transition_perf:.3f}',
                           xy=(self.transition_epoch, transition_perf),
                           xytext=(-50, 10), textcoords='offset points',
                           fontsize=8, ha='right')

                ax5.annotate(f'Final: {final_perf:.3f}\nImprovement: {improvement:.3f}',
                           xy=(len(epochs)-1, final_perf),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=8, ha='left')

            # Highlight best performance across all epochs
            best_epoch = np.argmin(self.target_val_loss)
            best_performance = self.target_val_loss[best_epoch]
            ax5.scatter([best_epoch], [best_performance], color='red', s=100,
                       zorder=5, label=f'Best: {best_performance:.3f} @ epoch {best_epoch}')

        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Target Validation Loss')
        ax5.set_title('Target Domain Performance (Evaluated from Epoch 0)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # Panel 6: Loss Distribution Analysis
        ax6 = axes[1, 2]

        # Create histogram of loss values to understand distribution
        all_losses = np.concatenate([self.full_train_loss, self.full_val_loss])

        ax6.hist(self.full_train_loss, bins=20, alpha=0.6, color=colors['train'],
                label='Train Loss Distribution', density=True)
        ax6.hist(self.full_val_loss, bins=20, alpha=0.6, color=colors['val'],
                label='Val Loss Distribution', density=True)

        # Add vertical lines for key statistics
        ax6.axvline(np.mean(self.full_train_loss), color=colors['train'],
                   linestyle='--', alpha=0.8, label='Train Mean')
        ax6.axvline(np.mean(self.full_val_loss), color=colors['val'],
                   linestyle='--', alpha=0.8, label='Val Mean')

        ax6.set_xlabel('Loss Value')
        ax6.set_ylabel('Density')
        ax6.set_title('Loss Distribution Analysis')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        # Panel 7: Convergence Analysis
        ax7 = axes[2, 0]

        if len(self.full_val_loss) > 10:
            # Rolling statistics to show convergence
            window = max(5, len(self.full_val_loss) // 10)

            # Rolling mean and std
            val_rolling_mean = pd.Series(self.full_val_loss).rolling(window=window).mean()
            val_rolling_std = pd.Series(self.full_val_loss).rolling(window=window).std()

            ax7.plot(self.epochs, val_rolling_mean, color=colors['val'],
                    linewidth=2, label=f'Rolling Mean (window={window})')
            ax7.fill_between(self.epochs,
                           val_rolling_mean - val_rolling_std,
                           val_rolling_mean + val_rolling_std,
                           alpha=0.3, color=colors['val'], label='±1 Std')

            # Mark convergence point (where std becomes small)
            if len(val_rolling_std.dropna()) > 0:
                convergence_threshold = np.percentile(val_rolling_std.dropna(), 25)
                converged_epochs = self.epochs[val_rolling_std <= convergence_threshold]
                if len(converged_epochs) > 0:
                    ax7.axvline(x=converged_epochs[0], color=colors['highlight'],
                               linestyle=':', label='Convergence Start')

        ax7.set_xlabel('Epoch')
        ax7.set_ylabel('Validation Loss')
        ax7.set_title('Convergence Analysis')
        ax7.legend()
        ax7.grid(True, alpha=0.3)

        # Panel 8: Training Efficiency Timeline
        ax8 = axes[2, 1]

        if len(self.full_train_loss) > 5:
            # Efficiency = improvement per epoch
            efficiency = []
            for i in range(1, len(self.full_train_loss)):
                recent_improvement = self.full_train_loss[max(0, i-5):i]
                if len(recent_improvement) > 1:
                    eff = (recent_improvement[0] - recent_improvement[-1]) / len(recent_improvement)
                    efficiency.append(max(0, eff))  # Only positive improvements
                else:
                    efficiency.append(0)

            efficiency_epochs = self.epochs[1:len(efficiency)+1]
            ax8.plot(efficiency_epochs, efficiency, color=colors['highlight'],
                    linewidth=2, label='Learning Efficiency')

            if self.transition_epoch > 0:
                ax8.axvline(x=self.transition_epoch, color=colors['transition'],
                           linestyle='--', alpha=0.8)

            # Smooth efficiency curve
            if len(efficiency) > 10:
                smooth_window = min(10, len(efficiency) // 5)
                efficiency_smooth = np.convolve(efficiency, np.ones(smooth_window)/smooth_window, mode='valid')
                smooth_epochs = efficiency_epochs[smooth_window//2:len(efficiency_smooth)+smooth_window//2]
                ax8.plot(smooth_epochs, efficiency_smooth, color=colors['train'],
                        linewidth=3, alpha=0.7, label='Smoothed')

        ax8.set_xlabel('Epoch')
        ax8.set_ylabel('Loss Improvement per Epoch')
        ax8.set_title('Learning Efficiency Timeline')
        ax8.legend()
        ax8.grid(True, alpha=0.3)

        # Panel 9: Summary Statistics and Health Score
        ax9 = axes[2, 2]
        ax9.axis('off')

        # Create summary text
        health_score = self.training_health.get('overall_health_score', 0)

        summary_text = f"📊 TRAINING SUMMARY\n\n"
        summary_text += f"Overall Health Score: {health_score:.1f}/100\n\n"

        # Key metrics
        summary_text += f"Key Metrics:\n"
        summary_text += f"• Overfitting Score: {self.training_health.get('overfitting_score', 0):.1f}/100\n"
        summary_text += f"• Learning Efficiency: {self.training_health.get('learning_efficiency', 0):.3f}\n"
        summary_text += f"• Convergence Quality: {self.training_health.get('convergence_quality', 0):.3f}\n"
        summary_text += f"• Training Smoothness: {self.training_health.get('overall_smoothness', 0):.3f}\n"

        if self.training_health.get('transfer_efficiency', 0) > 0:
            summary_text += f"• Transfer Efficiency: {self.training_health.get('transfer_efficiency', 0):.3f}\n"

        summary_text += f"\nTraining Details:\n"
        summary_text += f"• Total Epochs: {self.total_epochs}\n"
        summary_text += f"• Transition Epoch: {self.transition_epoch}\n"
        summary_text += f"• Final Train Loss: {self.training_health.get('final_train_loss', 0):.3f}\n"
        summary_text += f"• Final Val Loss: {self.training_health.get('final_val_loss', 0):.3f}\n"

        # Health assessment
        if health_score > 80:
            summary_text += f"\n✅ EXCELLENT HEALTH"
        elif health_score > 60:
            summary_text += f"\n👍 GOOD HEALTH"
        elif health_score > 40:
            summary_text += f"\n⚠️ MODERATE ISSUES"
        else:
            summary_text += f"\n❌ SIGNIFICANT PROBLEMS"

        ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

        # Adjust layout
        plt.tight_layout()

        # Save the dashboard
        if save_path is None:
            save_path = f"{self.run_dir}/training_dashboard.jpg"

        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

        return save_path

def analyze_single_run(experiment_path: str, run_id: str, save_results: bool = True) -> Dict[str, Any]:
    """Analyze a single training run and optionally save results."""
    analyzer = TrainingAnalyzer(experiment_path, run_id)
    results = analyzer.analyze_training()
    recommendations = analyzer.generate_recommendations()

    # Print summary
    print(f"\n🏥 TRAINING HEALTH ANALYSIS")
    print(f"Experiment: {experiment_path}")
    print(f"Run: {run_id}")
    print(f"Overall Health Score: {results['overall_health_score']:.1f}/100")

    print(f"\n📊 KEY METRICS:")
    print(f"• Overfitting Score: {results.get('overfitting_score', 0):.1f}/100")
    print(f"• Learning Efficiency: {results.get('learning_efficiency', 0):.3f}")
    print(f"• Convergence Quality: {results.get('convergence_quality', 0):.3f}")
    print(f"• Training Smoothness: {results.get('overall_smoothness', 0):.3f}")

    # Transfer learning specific metrics
    if results.get('transfer_gap', 0) != 0:  # We have transfer learning
        print(f"\n🔄 TRANSFER LEARNING METRICS:")
        print(f"• Transfer Gap: {results.get('transfer_gap', 0):.3f}")
        print(f"• Adaptation Speed: {results.get('adaptation_speed', 0):.3f}")
        print(f"• Knowledge Retention: {results.get('knowledge_retention', 0):.3f}")
        print(f"• Transfer Success Score: {results.get('transfer_success_score', 0):.3f}")
        print(f"• Target Improvement: {results.get('target_improvement_rate', 0):.3f}")
        print(f"• Catastrophic Forgetting: {results.get('catastrophic_forgetting', 0):.3f}")

    print(f"\n💡 RECOMMENDATIONS:")
    for rec in recommendations:
        print(rec)

    # Save detailed results
    if save_results:
        output_file = f"{experiment_path}/{run_id}/training_analysis.json"
        analysis_results = {
            'metrics': results,
            'recommendations': recommendations,
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }

        with open(output_file, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        print(f"\n💾 Detailed analysis saved to: {output_file}")

    return results

def main():
    parser = argparse.ArgumentParser(description='Analyze training dynamics and health')
    parser.add_argument('experiment', help='Experiment name (e.g., alpha, medium)')
    parser.add_argument('run_id', help='Run ID to analyze (e.g., 0025)')
    parser.add_argument('--no-save', action='store_true', help='Don\'t save analysis results')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization plots')

    args = parser.parse_args()

    experiment_path = f"experiments/{args.experiment}"

    if not os.path.exists(f"{experiment_path}/{args.run_id}"):
        print(f"Error: Training run not found: {experiment_path}/{args.run_id}")
        return

    # Run analysis
    results = analyze_single_run(experiment_path, args.run_id, not args.no_save)

    # Generate visualization if requested
    if args.visualize:
        analyzer = TrainingAnalyzer(experiment_path, args.run_id)
        analyzer.training_health = results  # Load computed results
        analyzer.create_training_dashboard()
        print(f"\n📈 Training visualization saved to: {experiment_path}/{args.run_id}/training_dashboard.jpg")

if __name__ == "__main__":
    main()