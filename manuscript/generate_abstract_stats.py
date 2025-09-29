#!/usr/bin/env python3
"""
Generate abstract statistics from combined target-only and full fine-tuning experiments.
"""
import os
import argparse
from generate_combined_results import load_experiment_results, extract_f1_scores

def generate_abstract_stats(target_only_exp: str, full_ft_exp: str, output_file: str = "abstract_stats.tex"):
    """Generate abstract statistics comparing target-only and full fine-tuning."""

    # Load both experiments
    target_only_dir = os.path.join('..', 'experiments', target_only_exp)
    full_ft_dir = os.path.join('..', 'experiments', full_ft_exp)

    target_only_results = load_experiment_results(target_only_dir)
    full_ft_results = load_experiment_results(full_ft_dir)

    # Extract F1 scores
    target_only_scores = extract_f1_scores(target_only_results, 'target_only')
    full_ft_scores = extract_f1_scores(full_ft_results, 'full_fine_tuning')

    # Extract base model scores
    base_model_scores = {}
    for participant, data in full_ft_results.items():
        metrics = data['metrics']
        base_model_scores[participant] = metrics.get('best_target_val_f1_from_best_base_model', 0.0)

    # Calculate improvements for common participants
    common_participants = set(target_only_scores.keys()) & set(full_ft_scores.keys()) & set(base_model_scores.keys())

    if common_participants:
        target_vs_base_improvements = []
        full_ft_vs_base_improvements = []
        full_ft_vs_target_improvements = []

        for p in common_participants:
            target_vs_base = (target_only_scores[p] - base_model_scores[p]) * 100
            full_ft_vs_base = (full_ft_scores[p] - base_model_scores[p]) * 100
            full_ft_vs_target = (full_ft_scores[p] - target_only_scores[p]) * 100

            target_vs_base_improvements.append(target_vs_base)
            full_ft_vs_base_improvements.append(full_ft_vs_base)
            full_ft_vs_target_improvements.append(full_ft_vs_target)

        mean_target_vs_base = sum(target_vs_base_improvements) / len(target_vs_base_improvements)
        mean_full_ft_vs_base = sum(full_ft_vs_base_improvements) / len(full_ft_vs_base_improvements)
        mean_full_ft_vs_target = sum(full_ft_vs_target_improvements) / len(full_ft_vs_target_improvements)

        min_target_gain = min(target_vs_base_improvements)
        max_target_gain = max(target_vs_base_improvements)
        min_full_ft_gain = min(full_ft_vs_base_improvements)
        max_full_ft_gain = max(full_ft_vs_base_improvements)

        full_ft_better_count = sum(1 for imp in full_ft_vs_target_improvements if imp > 0)
        target_better_count = len(common_participants) - full_ft_better_count

        # Generate LaTeX content
        latex_content = f"""% Auto-generated abstract statistics from combined experiments
% Target-Only: {target_only_exp}
% Full FT: {full_ft_exp}
% Generated on: {os.popen('date').read().strip()}

% Combined experiment statistics macros
\\newcommand{{\\numparticipants}}{{{len(common_participants)}}}
\\newcommand{{\\targetseparation}}{{{mean_target_vs_base:.1f}}}
\\newcommand{{\\fullftgain}}{{{mean_full_ft_vs_base:.1f}}}
\\newcommand{{\\ftadvantageovertarget}}{{{mean_full_ft_vs_target:.1f}}}
\\newcommand{{\\fullftsuccess}}{{{full_ft_better_count}}}
\\newcommand{{\\targetonlysuccess}}{{{target_better_count}}}

% Legacy macros for compatibility with existing abstract
\\newcommand{{\\mingain}}{{{min_full_ft_gain:.1f}}}
\\newcommand{{\\maxgain}}{{{max_full_ft_gain:.1f}}}
\\newcommand{{\\meangain}}{{{mean_full_ft_vs_base:.1f}}}
\\newcommand{{\\responderrate}}{{100}}

% Detailed statistics (for reference)
% Common participants: {sorted(list(common_participants))}
% Target-only vs base: {mean_target_vs_base:.1f}% (range: {min_target_gain:.1f}% to {max_target_gain:.1f}%)
% Full FT vs base: {mean_full_ft_vs_base:.1f}% (range: {min_full_ft_gain:.1f}% to {max_full_ft_gain:.1f}%)
% Full FT vs target-only: {mean_full_ft_vs_target:.1f}%
% Full FT better: {full_ft_better_count}/{len(common_participants)} participants
"""

        # Write to file
        with open(output_file, 'w') as f:
            f.write(latex_content)

        print(f"Generated {output_file} with combined abstract statistics")
        print(f"Target-only vs Base: {mean_target_vs_base:.1f}% improvement")
        print(f"Full FT vs Base: {mean_full_ft_vs_base:.1f}% improvement")
        print(f"Full FT vs Target-only: {mean_full_ft_vs_target:.1f}% improvement")
        print(f"Success rate: {full_ft_better_count}/{len(common_participants)} participants favor Full FT")

def main():
    parser = argparse.ArgumentParser(description='Generate abstract statistics from combined experiments')
    parser.add_argument('--target-only', required=True, help='Target-only experiment name')
    parser.add_argument('--full-ft', required=True, help='Full fine-tuning experiment name')
    parser.add_argument('--output', default='abstract_stats.tex', help='Output LaTeX file')

    args = parser.parse_args()

    generate_abstract_stats(args.target_only, args.full_ft, args.output)

if __name__ == "__main__":
    main()