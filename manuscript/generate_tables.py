#!/usr/bin/env python3
"""
Standalone script to generate individual LaTeX tables from experiment results.
Note: For full manuscript generation, use generate_results.py instead.
Usage: python generate_tables.py --experiment <experiment_name>
"""

import json
import os
import argparse
import glob
from typing import Dict, List, Tuple

def load_experiment_results(experiment_dir: str) -> Dict:
    """Load all fold results from an experiment directory."""
    results = {}

    # Find all fold directories
    fold_dirs = glob.glob(os.path.join(experiment_dir, "fold*"))

    for fold_dir in fold_dirs:
        fold_name = os.path.basename(fold_dir)

        # Extract participant name from fold directory
        participant = fold_name.split('_', 1)[1] if '_' in fold_name else fold_name

        # Load metrics and hyperparameters
        metrics_file = os.path.join(fold_dir, "metrics.json")
        hyperparam_file = os.path.join(fold_dir, "hyperparameters.json")

        if os.path.exists(metrics_file) and os.path.exists(hyperparam_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            with open(hyperparam_file, 'r') as f:
                hyperparams = json.load(f)

            results[participant] = {
                'metrics': metrics,
                'hyperparams': hyperparams,
                'fold_dir': fold_name
            }

    return results

def generate_performance_table(results: Dict, experiment_name: str) -> str:
    """Generate LaTeX table for performance comparison."""

    latex_lines = [
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{Transfer learning strategy comparison - {experiment_name}}}",
        "\\label{tab:results}",
        "\\begin{tabular}{@{}lcccc@{}}",
        "\\toprule",
        "Participant & Target-Only & Base Model & Full FT & Best Method \\\\",
        "\\midrule"
    ]

    total_base_f1 = 0
    total_target_f1 = 0
    count = 0

    for participant, data in sorted(results.items()):
        metrics = data['metrics']

        # Extract key metrics
        base_f1 = metrics.get('best_target_val_f1_from_best_base_model', 0)
        target_f1 = metrics.get('best_target_val_f1', 0)

        # Determine best method (simplified for now)
        best_method = "Full FT" if target_f1 > base_f1 else "Base Model"

        # Format row
        row = f"{participant} & [TBD] & {base_f1:.3f} & {target_f1:.3f} & {best_method} \\\\"
        latex_lines.append(row)

        total_base_f1 += base_f1
        total_target_f1 += target_f1
        count += 1

    # Add mean row
    if count > 0:
        mean_base = total_base_f1 / count
        mean_target = total_target_f1 / count
        latex_lines.extend([
            "\\midrule",
            f"Mean & [TBD] & {mean_base:.3f} & {mean_target:.3f} & - \\\\"
        ])

    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])

    return "\n".join(latex_lines)

def generate_detailed_metrics_table(results: Dict, experiment_name: str) -> str:
    """Generate detailed metrics table with improvements."""

    latex_lines = [
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{Detailed performance metrics - {experiment_name}}}",
        "\\label{tab:detailed_results}",
        "\\begin{tabular}{@{}lcccc@{}}",
        "\\toprule",
        "Participant & Base F1 & Full FT F1 & Improvement & \\% Improvement \\\\",
        "\\midrule"
    ]

    improvements = []

    for participant, data in sorted(results.items()):
        metrics = data['metrics']

        base_f1 = metrics.get('best_target_val_f1_from_best_base_model', 0)
        target_f1 = metrics.get('best_target_val_f1', 0)

        improvement = target_f1 - base_f1
        pct_improvement = (improvement / base_f1) * 100 if base_f1 > 0 else 0

        improvements.append(improvement)

        # Format improvement with sign
        imp_str = f"{improvement:+.3f}"
        pct_str = f"{pct_improvement:+.1f}\\%"

        row = f"{participant} & {base_f1:.3f} & {target_f1:.3f} & {imp_str} & {pct_str} \\\\"
        latex_lines.append(row)

    # Add mean row
    if improvements:
        mean_improvement = sum(improvements) / len(improvements)
        mean_base = sum(data['metrics'].get('best_target_val_f1_from_best_base_model', 0)
                       for data in results.values()) / len(results)
        mean_target = sum(data['metrics'].get('best_target_val_f1', 0)
                         for data in results.values()) / len(results)
        mean_pct = (mean_improvement / mean_base) * 100 if mean_base > 0 else 0

        latex_lines.extend([
            "\\midrule",
            f"Mean & {mean_base:.3f} & {mean_target:.3f} & {mean_improvement:+.3f} & {mean_pct:+.1f}\\% \\\\"
        ])

    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])

    return "\n".join(latex_lines)

def generate_training_dynamics_table(results: Dict, experiment_name: str) -> str:
    """Generate table for training dynamics analysis."""

    latex_lines = [
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{Training dynamics analysis - {experiment_name}}}",
        "\\label{tab:training_dynamics}",
        "\\begin{tabular}{@{}lccc@{}}",
        "\\toprule",
        "Participant & Transition Epoch & Best Base Epoch & Best Target Epoch \\\\",
        "\\midrule"
    ]

    for participant, data in sorted(results.items()):
        metrics = data['metrics']

        transition_epoch = metrics.get('transition_epoch', 'N/A')
        best_base_epoch = metrics.get('best_base_val_f1_epoch', 'N/A')
        best_target_epoch = metrics.get('best_target_val_f1_epoch', 'N/A')

        row = f"{participant} & {transition_epoch} & {best_base_epoch} & {best_target_epoch} \\\\"
        latex_lines.append(row)

    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])

    return "\n".join(latex_lines)

def main():
    parser = argparse.ArgumentParser(description='Generate LaTeX tables from experiment results')
    parser.add_argument('--experiment', required=True, help='Experiment directory name')
    parser.add_argument('--output', default='standalone_tables.tex', help='Output LaTeX file')
    parser.add_argument('--table', choices=['performance', 'detailed', 'training', 'all'],
                       default='all', help='Which table to generate')

    args = parser.parse_args()

    # Find experiment directory
    experiment_dir = os.path.join('..', 'experiments', args.experiment)
    if not os.path.exists(experiment_dir):
        print(f"Error: Experiment directory {experiment_dir} does not exist")
        return

    # Load results
    results = load_experiment_results(experiment_dir)
    if not results:
        print(f"Error: No results found in {experiment_dir}")
        return

    print(f"Found results for participants: {list(results.keys())}")

    # Generate tables
    tables = []

    if args.table in ['performance', 'all']:
        tables.append(generate_performance_table(results, args.experiment))

    if args.table in ['detailed', 'all']:
        tables.append(generate_detailed_metrics_table(results, args.experiment))

    if args.table in ['training', 'all']:
        tables.append(generate_training_dynamics_table(results, args.experiment))

    # Write output
    output_content = "\n\n".join(tables)

    with open(args.output, 'w') as f:
        f.write(output_content)

    print(f"Generated LaTeX tables saved to {args.output}")
    print("\nGenerated tables:")
    print(output_content)

if __name__ == "__main__":
    main()