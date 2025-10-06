"""
Statistical tests for base vs fine-tuned model comparison.
Performs paired t-tests and Wilcoxon signed-rank tests.
"""
import os
import json
import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm

# Load data same way as figure1.py
data = []
experiments_dir = '../experiments'

for experiment in tqdm(os.listdir(experiments_dir)):
    for run in os.listdir(f'{experiments_dir}/{experiment}'):
        if not os.path.exists(f'{experiments_dir}/{experiment}/{run}/metrics.json'):
            continue

        metrics = json.load(open(f'{experiments_dir}/{experiment}/{run}/metrics.json'))
        hyperparameters = json.load(open(f'{experiments_dir}/{experiment}/{run}/hyperparameters.json'))

        data.append({
            'hyperparameter_hash': experiment.split('_')[-1],
            'batch_size': hyperparameters['batch_size'],
            'fold': int(run.split('_')[0].replace('fold','')),
            'target_data_pct': float(hyperparameters['target_data_pct']),
            'n_base_participants': int(hyperparameters['n_base_participants']),
            'mode': hyperparameters['mode'],
            'best_target_model_target_val_f1': metrics['best_target_model_target_val_f1'],
            'best_target_model_target_test_f1': metrics['best_target_model_target_test_f1'],
            'best_base_model_target_val_f1': metrics['best_base_model_target_val_f1'] if 'best_base_model_target_val_f1' in metrics else None,
            'best_base_model_target_test_f1': metrics['best_base_model_target_test_f1'] if 'best_base_model_target_test_f1' in metrics else None,
            'absolute_improvement_target_test_f1': (metrics['best_target_model_target_test_f1'] - metrics['best_base_model_target_test_f1']) if 'best_base_model_target_test_f1' in metrics else None,
            'absolute_improvement_target_val_f1': (metrics['best_target_model_target_val_f1'] - metrics['best_base_model_target_val_f1']) if 'best_base_model_target_val_f1' in metrics else None,
        })

df = pd.DataFrame(data)

print("="*80)
print("STATISTICAL TESTS FOR BASE VS FINE-TUNED MODELS")
print("="*80)

# Filter for full_fine_tuning and target_only_fine_tuning modes, 100% target data
df_stats = df[(df['mode'].isin(['full_fine_tuning', 'target_only_fine_tuning']))].copy()

print(f"\nTotal samples: {len(df_stats)}")
print(f"Unique folds: {df_stats['fold'].nunique()}")
print(f"Modes: {df_stats['mode'].unique()}")

# ==============================================================================
# TEST 1: Overall paired comparison (all hyperparameter configurations)
# ==============================================================================
print("\n" + "="*80)
print("TEST 1: Overall Paired Comparison (All Hyperparameter Configurations)")
print("="*80)

# Remove rows where base model F1 is missing
df_paired = df_stats[df_stats['best_base_model_target_test_f1'].notna()].copy()

base_f1 = df_paired['best_base_model_target_test_f1'].values
finetuned_f1 = df_paired['best_target_model_target_test_f1'].values

# Paired t-test
t_stat, p_value_ttest = stats.ttest_rel(finetuned_f1, base_f1)
print(f"\nPaired t-test (fine-tuned vs base):")
print(f"  Sample size: {len(base_f1)}")
print(f"  Base F1: {base_f1.mean():.4f} ± {base_f1.std():.4f}")
print(f"  Fine-tuned F1: {finetuned_f1.mean():.4f} ± {finetuned_f1.std():.4f}")
print(f"  Mean difference: {(finetuned_f1 - base_f1).mean():.4f} ± {(finetuned_f1 - base_f1).std():.4f}")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value_ttest:.6f}")
print(f"  Significant at α=0.05: {'Yes' if p_value_ttest < 0.05 else 'No'}")
print(f"  Significant at α=0.01: {'Yes' if p_value_ttest < 0.01 else 'No'}")

# Wilcoxon signed-rank test (non-parametric)
w_stat, p_value_wilcoxon = stats.wilcoxon(finetuned_f1, base_f1)
print(f"\nWilcoxon signed-rank test (fine-tuned vs base):")
print(f"  W-statistic: {w_stat:.4f}")
print(f"  p-value: {p_value_wilcoxon:.6f}")
print(f"  Significant at α=0.05: {'Yes' if p_value_wilcoxon < 0.05 else 'No'}")
print(f"  Significant at α=0.01: {'Yes' if p_value_wilcoxon < 0.01 else 'No'}")

# Effect size (Cohen's d for paired samples)
diff = finetuned_f1 - base_f1
cohens_d = diff.mean() / diff.std()
print(f"\nEffect size (Cohen's d): {cohens_d:.4f}")
print(f"  Interpretation: ", end="")
if abs(cohens_d) < 0.2:
    print("negligible")
elif abs(cohens_d) < 0.5:
    print("small")
elif abs(cohens_d) < 0.8:
    print("medium")
else:
    print("large")

# ==============================================================================
# TEST 2: Per-fold paired comparison (averaging across hyperparameters per fold)
# ==============================================================================
print("\n" + "="*80)
print("TEST 2: Per-Fold Paired Comparison (Median Performance per Fold)")
print("="*80)

# For each fold, compute median performance across all hyperparameter configurations
fold_summary = df_paired.groupby('fold').agg({
    'best_base_model_target_test_f1': 'median',
    'best_target_model_target_test_f1': 'median',
    'absolute_improvement_target_test_f1': 'median'
}).reset_index()

print(f"\nFold-level summary (median across hyperparameters):")
print(fold_summary.to_string(index=False))

base_f1_per_fold = fold_summary['best_base_model_target_test_f1'].values
finetuned_f1_per_fold = fold_summary['best_target_model_target_test_f1'].values

# Paired t-test on fold-level medians
t_stat_fold, p_value_ttest_fold = stats.ttest_rel(finetuned_f1_per_fold, base_f1_per_fold)
print(f"\nPaired t-test on fold-level medians (fine-tuned vs base):")
print(f"  Number of folds: {len(base_f1_per_fold)}")
print(f"  Base F1: {base_f1_per_fold.mean():.4f} ± {base_f1_per_fold.std():.4f}")
print(f"  Fine-tuned F1: {finetuned_f1_per_fold.mean():.4f} ± {finetuned_f1_per_fold.std():.4f}")
print(f"  Mean difference: {(finetuned_f1_per_fold - base_f1_per_fold).mean():.4f} ± {(finetuned_f1_per_fold - base_f1_per_fold).std():.4f}")
print(f"  t-statistic: {t_stat_fold:.4f}")
print(f"  p-value: {p_value_ttest_fold:.6f}")
print(f"  Significant at α=0.05: {'Yes' if p_value_ttest_fold < 0.05 else 'No'}")
print(f"  Significant at α=0.01: {'Yes' if p_value_ttest_fold < 0.01 else 'No'}")

# Wilcoxon signed-rank test on fold-level medians
w_stat_fold, p_value_wilcoxon_fold = stats.wilcoxon(finetuned_f1_per_fold, base_f1_per_fold)
print(f"\nWilcoxon signed-rank test on fold-level medians (fine-tuned vs base):")
print(f"  W-statistic: {w_stat_fold:.4f}")
print(f"  p-value: {p_value_wilcoxon_fold:.6f}")
print(f"  Significant at α=0.05: {'Yes' if p_value_wilcoxon_fold < 0.05 else 'No'}")
print(f"  Significant at α=0.01: {'Yes' if p_value_wilcoxon_fold < 0.01 else 'No'}")

# Effect size
diff_fold = finetuned_f1_per_fold - base_f1_per_fold
cohens_d_fold = diff_fold.mean() / diff_fold.std()
print(f"\nEffect size (Cohen's d): {cohens_d_fold:.4f}")

# ==============================================================================
# TEST 3: Individual fold t-tests
# ==============================================================================
print("\n" + "="*80)
print("TEST 3: Individual Fold t-tests (Within Each Fold)")
print("="*80)

fold_test_results = []
for fold in sorted(df_paired['fold'].unique()):
    df_fold = df_paired[df_paired['fold'] == fold]
    base = df_fold['best_base_model_target_test_f1'].values
    finetuned = df_fold['best_target_model_target_test_f1'].values

    if len(base) > 1:  # Need at least 2 samples for t-test
        t_stat_i, p_value_i = stats.ttest_rel(finetuned, base)
        w_stat_i, p_wilcox_i = stats.wilcoxon(finetuned, base)

        fold_test_results.append({
            'fold': fold,
            'n': len(base),
            'base_mean': base.mean(),
            'finetuned_mean': finetuned.mean(),
            'mean_diff': (finetuned - base).mean(),
            't_stat': t_stat_i,
            'p_ttest': p_value_i,
            'p_wilcoxon': p_wilcox_i,
            'sig_ttest_005': p_value_i < 0.05,
            'sig_wilcoxon_005': p_wilcox_i < 0.05
        })

df_fold_tests = pd.DataFrame(fold_test_results)
print("\nIndividual fold tests:")
print(df_fold_tests.to_string(index=False))

# ==============================================================================
# TEST 4: Best hyperparameter configuration analysis
# ==============================================================================
print("\n" + "="*80)
print("TEST 4: Best Hyperparameter Configuration (From Table 1)")
print("="*80)

# Filter for 100% target data, N=6 base participants
# This matches Table 1 in the manuscript
df_best = df_stats[
    (df_stats['target_data_pct'] == 1.0) &
    (df_stats['n_base_participants'] == 6)
].copy()

if len(df_best) > 0:
    # Group by fold and take median across batch sizes
    fold_best = df_best.groupby('fold').agg({
        'best_base_model_target_test_f1': 'median',
        'best_target_model_target_test_f1': 'median',
        'absolute_improvement_target_test_f1': 'median'
    }).reset_index()

    print(f"\nConfiguration: 100% target data, N=6 base participants")
    print(fold_best.to_string(index=False))

    base_best = fold_best['best_base_model_target_test_f1'].values
    finetuned_best = fold_best['best_target_model_target_test_f1'].values

    t_stat_best, p_value_best = stats.ttest_rel(finetuned_best, base_best)
    w_stat_best, p_wilcox_best = stats.wilcoxon(finetuned_best, base_best)

    print(f"\nPaired t-test (N={len(base_best)} folds):")
    print(f"  Base F1: {base_best.mean():.4f} ± {base_best.std():.4f}")
    print(f"  Fine-tuned F1: {finetuned_best.mean():.4f} ± {finetuned_best.std():.4f}")
    print(f"  Mean difference: {(finetuned_best - base_best).mean():.4f} ± {(finetuned_best - base_best).std():.4f}")
    print(f"  t-statistic: {t_stat_best:.4f}")
    print(f"  p-value: {p_value_best:.6f}")
    print(f"  Significant at α=0.05: {'Yes' if p_value_best < 0.05 else 'No'}")

    print(f"\nWilcoxon signed-rank test:")
    print(f"  W-statistic: {w_stat_best:.4f}")
    print(f"  p-value: {p_wilcox_best:.6f}")
    print(f"  Significant at α=0.05: {'Yes' if p_wilcox_best < 0.05 else 'No'}")

    # One-sample t-test on improvements (test if mean improvement > 0)
    improvements = finetuned_best - base_best
    t_stat_imp, p_value_imp = stats.ttest_1samp(improvements, 0)
    print(f"\nOne-sample t-test (improvement > 0):")
    print(f"  Mean improvement: {improvements.mean():.4f}")
    print(f"  t-statistic: {t_stat_imp:.4f}")
    print(f"  p-value (one-tailed): {p_value_imp/2:.6f}")
    print(f"  Significant at α=0.05: {'Yes' if p_value_imp/2 < 0.05 else 'No'}")

# ==============================================================================
# Save results to text file
# ==============================================================================
print("\n" + "="*80)
print("Saving results to statistical_tests_results.txt")
print("="*80)

with open('statistical_tests_results.txt', 'w') as f:
    f.write("STATISTICAL TESTS FOR BASE VS FINE-TUNED MODELS\n")
    f.write("="*80 + "\n\n")

    f.write("TEST 1: Overall Paired Comparison\n")
    f.write(f"  Paired t-test: t={t_stat:.4f}, p={p_value_ttest:.6f}\n")
    f.write(f"  Wilcoxon test: W={w_stat:.4f}, p={p_value_wilcoxon:.6f}\n")
    f.write(f"  Cohen's d: {cohens_d:.4f}\n\n")

    f.write("TEST 2: Per-Fold Paired Comparison (Median per Fold)\n")
    f.write(f"  Paired t-test: t={t_stat_fold:.4f}, p={p_value_ttest_fold:.6f}\n")
    f.write(f"  Wilcoxon test: W={w_stat_fold:.4f}, p={p_value_wilcoxon_fold:.6f}\n")
    f.write(f"  Cohen's d: {cohens_d_fold:.4f}\n\n")

    if len(df_best) > 0:
        f.write("TEST 4: Best Configuration (100% data, N=6 base participants)\n")
        f.write(f"  Paired t-test: t={t_stat_best:.4f}, p={p_value_best:.6f}\n")
        f.write(f"  Wilcoxon test: W={w_stat_best:.4f}, p={p_wilcox_best:.6f}\n")
        f.write(f"  One-sample t-test (improvement > 0): t={t_stat_imp:.4f}, p={p_value_imp/2:.6f}\n")

print("\nResults saved successfully!")
