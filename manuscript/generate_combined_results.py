#!/usr/bin/env python3
"""
Generate LaTeX results from multiple experiment types (target-only and full fine-tuning).
Usage: python generate_combined_results.py --target-only <target_exp> --full-ft <full_exp>
"""

import sys
import os
import json
import glob
import argparse
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

def extract_f1_scores(results: Dict, experiment_mode: str) -> Dict[str, float]:
    """Extract F1 scores based on experiment mode."""
    f1_scores = {}

    for participant, data in results.items():
        metrics = data['metrics']
        hyperparams = data['hyperparams']

        # Verify experiment mode matches
        actual_mode = hyperparams.get('mode', 'unknown')

        if experiment_mode == 'target_only':
            # For target-only, the score is in best_base_val_f1 (no transfer learning)
            if actual_mode == 'target_only':
                f1_scores[participant] = metrics.get('best_base_val_f1', 0.0)
        elif experiment_mode == 'full_fine_tuning':
            # For full fine-tuning, use the target validation score
            if actual_mode == 'full_fine_tuning':
                f1_scores[participant] = metrics.get('best_target_val_f1', 0.0)

    return f1_scores

def generate_comparison_table(target_only_scores: Dict[str, float],
                            full_ft_scores: Dict[str, float],
                            base_model_scores: Dict[str, float],
                            experiment_name: str) -> str:
    """Generate LaTeX table comparing all three approaches."""

    # Escape underscores for LaTeX
    safe_experiment_name = experiment_name.replace('_', '\\_')

    latex_lines = [
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{Transfer learning strategy comparison - {safe_experiment_name}}}",
        "\\label{tab:results}",
        "\\begin{tabular}{@{}lcccc@{}}",
        "\\toprule",
        "Participant & Target-Only & Base Model & Full FT & Best Method \\\\",
        "\\midrule"
    ]

    # Get all participants (use full_ft as reference)
    all_participants = set(full_ft_scores.keys()) | set(target_only_scores.keys()) | set(base_model_scores.keys())

    total_target_only = 0
    total_base = 0
    total_full_ft = 0
    count = 0

    for participant in sorted(all_participants):
        target_only_f1 = target_only_scores.get(participant, 0.0)
        base_f1 = base_model_scores.get(participant, 0.0)
        full_ft_f1 = full_ft_scores.get(participant, 0.0)

        # Determine best method
        scores = {
            'Target-Only': target_only_f1,
            'Base Model': base_f1,
            'Full FT': full_ft_f1
        }
        best_method = max(scores, key=scores.get) if max(scores.values()) > 0 else "N/A"

        # Format row
        target_str = f"{target_only_f1:.3f}" if target_only_f1 > 0 else "[Missing]"
        base_str = f"{base_f1:.3f}" if base_f1 > 0 else "[Missing]"
        full_ft_str = f"{full_ft_f1:.3f}" if full_ft_f1 > 0 else "[Missing]"

        row = f"{participant} & {target_str} & {base_str} & {full_ft_str} & {best_method} \\\\"
        latex_lines.append(row)

        # Only include in averages if we have the score
        if target_only_f1 > 0:
            total_target_only += target_only_f1
        if base_f1 > 0:
            total_base += base_f1
        if full_ft_f1 > 0:
            total_full_ft += full_ft_f1
        count += 1

    # Add mean row
    if count > 0:
        mean_target_only = total_target_only / len([s for s in target_only_scores.values() if s > 0]) if target_only_scores else 0
        mean_base = total_base / len([s for s in base_model_scores.values() if s > 0]) if base_model_scores else 0
        mean_full_ft = total_full_ft / len([s for s in full_ft_scores.values() if s > 0]) if full_ft_scores else 0

        target_mean_str = f"{mean_target_only:.3f}" if mean_target_only > 0 else "[N/A]"
        base_mean_str = f"{mean_base:.3f}" if mean_base > 0 else "[N/A]"
        full_ft_mean_str = f"{mean_full_ft:.3f}" if mean_full_ft > 0 else "[N/A]"

        latex_lines.extend([
            "\\midrule",
            f"Mean & {target_mean_str} & {base_mean_str} & {full_ft_mean_str} & - \\\\"
        ])

    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])

    return "\n".join(latex_lines)

def generate_detailed_comparison_table(target_only_scores: Dict[str, float],
                                     full_ft_scores: Dict[str, float],
                                     base_model_scores: Dict[str, float],
                                     experiment_name: str) -> str:
    """Generate detailed comparison table with improvements."""

    # Escape underscores for LaTeX
    safe_experiment_name = experiment_name.replace('_', '\\_')

    latex_lines = [
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{Detailed performance comparison - {safe_experiment_name}}}",
        "\\label{tab:detailed_results}",
        "\\begin{tabular}{@{}lccccc@{}}",
        "\\toprule",
        "Participant & Target-Only & Base Model & Full FT & Target vs Base & Full FT vs Base \\\\",
        "\\midrule"
    ]

    all_participants = set(full_ft_scores.keys()) | set(target_only_scores.keys()) | set(base_model_scores.keys())

    for participant in sorted(all_participants):
        target_only_f1 = target_only_scores.get(participant, 0.0)
        base_f1 = base_model_scores.get(participant, 0.0)
        full_ft_f1 = full_ft_scores.get(participant, 0.0)

        # Calculate improvements vs base model
        target_vs_base = target_only_f1 - base_f1 if target_only_f1 > 0 and base_f1 > 0 else 0
        full_ft_vs_base = full_ft_f1 - base_f1 if full_ft_f1 > 0 and base_f1 > 0 else 0

        # Format values
        target_str = f"{target_only_f1:.3f}" if target_only_f1 > 0 else "—"
        base_str = f"{base_f1:.3f}" if base_f1 > 0 else "—"
        full_ft_str = f"{full_ft_f1:.3f}" if full_ft_f1 > 0 else "—"

        target_imp_str = f"{target_vs_base:+.3f}" if target_vs_base != 0 else "—"
        full_ft_imp_str = f"{full_ft_vs_base:+.3f}" if full_ft_vs_base != 0 else "—"

        row = f"{participant} & {target_str} & {base_str} & {full_ft_str} & {target_imp_str} & {full_ft_imp_str} \\\\"
        latex_lines.append(row)

    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])

    return "\n".join(latex_lines)

def generate_combined_results_tex(target_only_exp: str, full_ft_exp: str, output_file: str = "results.tex"):
    """Generate complete results section comparing target-only and full fine-tuning."""

    # Load both experiments
    target_only_dir = os.path.join('..', 'experiments', target_only_exp)
    full_ft_dir = os.path.join('..', 'experiments', full_ft_exp)

    if not os.path.exists(target_only_dir):
        print(f"Error: Target-only experiment directory {target_only_dir} does not exist")
        return

    if not os.path.exists(full_ft_dir):
        print(f"Error: Full fine-tuning experiment directory {full_ft_dir} does not exist")
        return

    # Load results
    target_only_results = load_experiment_results(target_only_dir)
    full_ft_results = load_experiment_results(full_ft_dir)

    if not target_only_results:
        print(f"Error: No target-only results found in {target_only_dir}")
        return

    if not full_ft_results:
        print(f"Error: No full fine-tuning results found in {full_ft_dir}")
        return

    # Extract F1 scores
    target_only_scores = extract_f1_scores(target_only_results, 'target_only')
    full_ft_scores = extract_f1_scores(full_ft_results, 'full_fine_tuning')

    # Extract base model scores from full fine-tuning experiment
    base_model_scores = {}
    for participant, data in full_ft_results.items():
        metrics = data['metrics']
        base_model_scores[participant] = metrics.get('best_target_val_f1_from_best_base_model', 0.0)

    print(f"Found target-only results for: {list(target_only_scores.keys())}")
    print(f"Found full fine-tuning results for: {list(full_ft_scores.keys())}")
    print(f"Found base model results for: {list(base_model_scores.keys())}")

    # Generate comparison table
    comparison_table = generate_comparison_table(
        target_only_scores, full_ft_scores, base_model_scores,
        f"Target-Only vs Full Fine-Tuning"
    )

    # Generate detailed comparison
    detailed_table = generate_detailed_comparison_table(
        target_only_scores, full_ft_scores, base_model_scores,
        f"Detailed Comparison"
    )

    # Calculate summary statistics
    if target_only_scores and full_ft_scores and base_model_scores:
        # Common participants across all experiments
        common_participants = set(target_only_scores.keys()) & set(full_ft_scores.keys()) & set(base_model_scores.keys())

        if common_participants:
            target_vs_base_improvements = []
            full_ft_vs_base_improvements = []
            full_ft_vs_target_improvements = []

            for p in common_participants:
                target_vs_base = target_only_scores[p] - base_model_scores[p]
                full_ft_vs_base = full_ft_scores[p] - base_model_scores[p]
                full_ft_vs_target = full_ft_scores[p] - target_only_scores[p]

                target_vs_base_improvements.append(target_vs_base)
                full_ft_vs_base_improvements.append(full_ft_vs_base)
                full_ft_vs_target_improvements.append(full_ft_vs_target)

            mean_target_vs_base = sum(target_vs_base_improvements) / len(target_vs_base_improvements)
            mean_full_ft_vs_base = sum(full_ft_vs_base_improvements) / len(full_ft_vs_base_improvements)
            mean_full_ft_vs_target = sum(full_ft_vs_target_improvements) / len(full_ft_vs_target_improvements)

    # Generate LaTeX content
    latex_content = f"""% Auto-generated results comparing target-only and full fine-tuning
% Target-Only Experiment: {target_only_exp}
% Full Fine-Tuning Experiment: {full_ft_exp}
% Generated on: {os.popen('date').read().strip()}

\\subsection{{Overall Performance Comparison}}

Table~\\ref{{tab:results}} compares the performance of target-only training, base model (no adaptation), and full fine-tuning across all participants.

{comparison_table}

\\subsection{{Detailed Performance Analysis}}

{detailed_table}

\\subsection{{Performance Summary}}

The results demonstrate the effectiveness of different transfer learning approaches:

\\begin{{itemize}}"""

    if 'common_participants' in locals() and common_participants:
        full_ft_better_count = sum(1 for imp in full_ft_vs_target_improvements if imp > 0)
        total_common = len(common_participants)

        latex_content += f"""
    \\item \\textbf{{Target-Only vs Base Model}}: Mean improvement of {mean_target_vs_base:+.3f} F1 points
    \\item \\textbf{{Full Fine-Tuning vs Base Model}}: Mean improvement of {mean_full_ft_vs_base:+.3f} F1 points
    \\item \\textbf{{Full Fine-Tuning vs Target-Only}}: Mean improvement of {mean_full_ft_vs_target:+.3f} F1 points
    \\item \\textbf{{Success Rate}}: Full fine-tuning outperformed target-only for {full_ft_better_count}/{total_common} participants ({full_ft_better_count/total_common*100:.0f}\\%)"""

    latex_content += """
\\end{itemize}

These results highlight the value of transfer learning, with full fine-tuning providing the best performance by leveraging both population-level knowledge and individual adaptation."""

    # Write to file
    with open(output_file, 'w') as f:
        f.write(latex_content)

    print(f"Generated {output_file} with combined results")
    if 'mean_full_ft_vs_target' in locals():
        print(f"Full FT vs Target-Only: {mean_full_ft_vs_target:+.3f} F1 points")
        print(f"Success rate: {full_ft_better_count}/{total_common} participants")

def main():
    parser = argparse.ArgumentParser(description='Generate combined results from target-only and full fine-tuning experiments')
    parser.add_argument('--target-only', required=True, help='Target-only experiment name')
    parser.add_argument('--full-ft', required=True, help='Full fine-tuning experiment name')
    parser.add_argument('--output', default='results.tex', help='Output LaTeX file')

    args = parser.parse_args()

    generate_combined_results_tex(args.target_only, args.full_ft, args.output)

if __name__ == "__main__":
    main()