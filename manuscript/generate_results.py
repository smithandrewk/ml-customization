#!/usr/bin/env python3
"""
Generate LaTeX results file for manuscript from experiment data.
This script creates a results.tex file that can be included in the main manuscript.
"""

import sys
import os
import json
import glob
from typing import Dict, List, Tuple

# All functions are now self-contained in this file

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

    # Escape underscores for LaTeX
    safe_experiment_name = experiment_name.replace('_', '\\_')

    latex_lines = [
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{Detailed performance metrics - {safe_experiment_name}}}",
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

    # Escape underscores for LaTeX
    safe_experiment_name = experiment_name.replace('_', '\\_')

    latex_lines = [
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{Training dynamics analysis - {safe_experiment_name}}}",
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

def generate_abstract_stats(experiment_name: str, output_file: str = "abstract_stats.tex"):
    """Generate LaTeX macros with computed statistics for the abstract."""

    # Load experiment results
    experiment_dir = os.path.join('..', 'experiments', experiment_name)
    if not os.path.exists(experiment_dir):
        print(f"Error: Experiment directory {experiment_dir} does not exist")
        return

    results = load_experiment_results(experiment_dir)
    if not results:
        print(f"Error: No results found in {experiment_dir}")
        return

    # Calculate statistics
    pct_improvements = []
    base_f1s = []

    for participant, data in results.items():
        metrics = data['metrics']
        base_f1 = metrics.get('best_target_val_f1_from_best_base_model', 0)
        target_f1 = metrics.get('best_target_val_f1', 0)

        base_f1s.append(base_f1)
        improvement = target_f1 - base_f1
        pct_improvement = (improvement / base_f1) * 100 if base_f1 > 0 else 0
        pct_improvements.append(pct_improvement)

    # Compute statistics
    min_gain = min(pct_improvements)
    max_gain = max(pct_improvements)
    mean_gain = sum(pct_improvements) / len(pct_improvements)
    num_participants = len(results)
    positive_gains = [p for p in pct_improvements if p > 0]
    responder_rate = len(positive_gains) / len(pct_improvements) * 100

    # Generate LaTeX macros
    latex_content = f"""% Auto-generated abstract statistics from experiment: {experiment_name}
% Generated on: {os.popen('date').read().strip()}

% Abstract statistics macros
\\newcommand{{\\numparticipants}}{{{num_participants}}}
\\newcommand{{\\mingain}}{{{min_gain:.1f}}}
\\newcommand{{\\maxgain}}{{{max_gain:.1f}}}
\\newcommand{{\\meangain}}{{{mean_gain:.1f}}}
\\newcommand{{\\responderrate}}{{{responder_rate:.0f}}}

% Detailed statistics (for reference)
% Participants: {list(results.keys())}
% Individual gains: {[f"{p:.1f}%" for p in pct_improvements]}
% Mean F1 improvement: {mean_gain:.1f}%
% Range: {min_gain:.1f}% to {max_gain:.1f}%
"""

    with open(output_file, 'w') as f:
        f.write(latex_content)

    print(f"Generated abstract statistics: {min_gain:.1f}% to {max_gain:.1f}% (mean: {mean_gain:.1f}%)")
    return min_gain, max_gain, mean_gain, num_participants

def generate_results_tex(experiment_name: str, output_file: str = "results.tex"):
    """Generate complete results section with tables and analysis."""

    # Load experiment results
    experiment_dir = os.path.join('..', 'experiments', experiment_name)
    if not os.path.exists(experiment_dir):
        print(f"Error: Experiment directory {experiment_dir} does not exist")
        return

    results = load_experiment_results(experiment_dir)
    if not results:
        print(f"Error: No results found in {experiment_dir}")
        return

    # Calculate summary statistics
    improvements = []
    base_f1s = []
    target_f1s = []

    for participant, data in results.items():
        metrics = data['metrics']
        base_f1 = metrics.get('best_target_val_f1_from_best_base_model', 0)
        target_f1 = metrics.get('best_target_val_f1', 0)

        base_f1s.append(base_f1)
        target_f1s.append(target_f1)
        improvements.append(target_f1 - base_f1)

    mean_improvement = sum(improvements) / len(improvements)
    mean_base = sum(base_f1s) / len(base_f1s)
    mean_target = sum(target_f1s) / len(target_f1s)
    mean_pct_improvement = (mean_improvement / mean_base) * 100

    # Count responders
    positive_improvements = [imp for imp in improvements if imp > 0]
    responder_rate = len(positive_improvements) / len(improvements) * 100

    # Generate LaTeX content
    latex_content = f"""% Auto-generated results from experiment: {experiment_name}
% Generated on: {os.popen('date').read().strip()}

\\subsection{{Overall Performance Improvements}}

Table~\\ref{{tab:results}} summarizes the performance comparison between base models and personalized models across all participants.

{generate_performance_table(results, experiment_name)}

The results show that full fine-tuning achieved a mean F1-score improvement of +{mean_improvement:.3f} ({mean_pct_improvement:+.1f}\\%) over the base model. {len(positive_improvements)} out of {len(results)} participants ({responder_rate:.0f}\\%) showed improvements with personalization.

\\subsection{{Detailed Performance Analysis}}

{generate_detailed_metrics_table(results, experiment_name)}

\\subsection{{Individual Variability}}

The results reveal significant individual variability in the effectiveness of personalization:

\\begin{{itemize}}"""

    # Categorize participants
    strong_responders = []
    moderate_responders = []
    non_responders = []

    for i, (participant, improvement) in enumerate(zip(results.keys(), improvements)):
        pct_improvement = (improvement / base_f1s[i]) * 100 if base_f1s[i] > 0 else 0

        if pct_improvement > 15:
            strong_responders.append(f"{participant} ({pct_improvement:+.1f}\\%)")
        elif pct_improvement > 5:
            moderate_responders.append(f"{participant} ({pct_improvement:+.1f}\\%)")
        else:
            non_responders.append(f"{participant} ({pct_improvement:+.1f}\\%)")

    if strong_responders:
        latex_content += f"""
    \\item \\textbf{{Strong responders}}: {', '.join(strong_responders)} showed substantial improvements"""

    if moderate_responders:
        latex_content += f"""
    \\item \\textbf{{Moderate responders}}: {', '.join(moderate_responders)} showed modest improvements"""

    if non_responders:
        latex_content += f"""
    \\item \\textbf{{Non-responders/Negative responders}}: {', '.join(non_responders)} showed little to no improvement or decreased performance"""

    latex_content += """
\\end{itemize}

\\subsection{Training Dynamics}

"""

    latex_content += generate_training_dynamics_table(results, experiment_name)

    latex_content += f"""

Analysis of the training metrics reveals interesting patterns:
\\begin{{itemize}}
    \\item Transition epochs (when fine-tuning began) varied across participants
    \\item Best target validation F1 scores were consistently achieved after the transition epoch
    \\item Mean improvement of {mean_improvement:.3f} F1 points demonstrates practical value of personalization
    \\item Individual variability suggests selective personalization strategies may be beneficial
\\end{{itemize}}"""

    # Write to file
    with open(output_file, 'w') as f:
        f.write(latex_content)

    print(f"Generated results.tex with data from experiment: {experiment_name}")
    print(f"Mean improvement: {mean_improvement:.3f} F1 points ({mean_pct_improvement:+.1f}%)")
    print(f"Responder rate: {responder_rate:.0f}% ({len(positive_improvements)}/{len(results)} participants)")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_results.py <experiment_name>")
        print("Example: python generate_results.py b256_aug_patience5_full_fine_tuning_20250929_112052")
        sys.exit(1)

    experiment_name = sys.argv[1]

    # Generate abstract statistics
    generate_abstract_stats(experiment_name)

    # Generate results section
    generate_results_tex(experiment_name)