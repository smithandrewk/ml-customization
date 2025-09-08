#!/usr/bin/env python3
"""
Statistical Analysis of LOPO Experiment Results
==============================================

Analyzes Leave-One-Participant-Out (LOPO) customization experiments to determine:
- Statistical significance of personalization improvements
- Effect sizes and confidence intervals
- Individual participant patterns and outliers
- Publication-ready statistical summaries

Usage:
    python scripts/analyze_results.py --results_dir results/lopo_6participants
    
Output:
    - results/statistical_analysis.json - Detailed statistical results
    - results/statistical_summary.txt - Human-readable report
    - results/participant_analysis.csv - Per-participant breakdown
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ttest_rel, shapiro, wilcoxon
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

def calculate_cohens_d(x, y):
    """Calculate Cohen's d effect size for paired samples."""
    diff = x - y
    pooled_std = np.sqrt(((len(x) - 1) * np.var(x, ddof=1) + (len(y) - 1) * np.var(y, ddof=1)) / (len(x) + len(y) - 2))
    if pooled_std == 0:
        return 0.0
    return np.mean(diff) / pooled_std

def bootstrap_confidence_interval(data, stat_func, n_bootstrap=10000, confidence=0.95):
    """Calculate bootstrap confidence interval for a statistic."""
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_stats.append(stat_func(bootstrap_sample))
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    return lower, upper

def load_lopo_results(results_dir):
    """Load LOPO experiment results from directory."""
    summary_file = os.path.join(results_dir, 'lopo_summary.csv')
    
    if not os.path.exists(summary_file):
        raise FileNotFoundError(f"LOPO summary file not found: {summary_file}")
    
    df = pd.read_csv(summary_file)
    
    # Filter only successful experiments with performance metrics
    successful_df = df[
        (df['status'] == 'success') & 
        df['base_f1'].notna() & 
        df['custom_f1'].notna()
    ].copy()
    
    if len(successful_df) == 0:
        raise ValueError("No successful experiments with performance metrics found")
    
    print(f"Loaded {len(successful_df)} successful experiments from {len(df)} total")
    return successful_df

def test_normality(data, name="data"):
    """Test if data is normally distributed using Shapiro-Wilk test."""
    if len(data) < 3:
        return False, 1.0, "insufficient data"
    
    stat, p_value = shapiro(data)
    is_normal = p_value > 0.05
    
    return is_normal, p_value, f"{name}: W={stat:.4f}, p={p_value:.4f}"

def perform_statistical_tests(df):
    """Perform comprehensive statistical analysis on LOPO results."""
    results = {
        'timestamp': datetime.now().isoformat(),
        'n_participants': len(df),
        'participants': df['participant'].tolist()
    }
    
    # Extract performance data
    base_f1 = df['base_f1'].values
    custom_f1 = df['custom_f1'].values
    improvements = df['f1_improvement'].values
    
    print(f"\nPerformance Summary:")
    print(f"  Base F1:    {np.mean(base_f1):.4f} ± {np.std(base_f1):.4f} (range: {np.min(base_f1):.4f}-{np.max(base_f1):.4f})")
    print(f"  Custom F1:  {np.mean(custom_f1):.4f} ± {np.std(custom_f1):.4f} (range: {np.min(custom_f1):.4f}-{np.max(custom_f1):.4f})")
    print(f"  Improvement: {np.mean(improvements):.4f} ± {np.std(improvements):.4f} (range: {np.min(improvements):.4f}-{np.max(improvements):.4f})")
    
    # Basic descriptive statistics
    results['descriptive_stats'] = {
        'base_f1': {
            'mean': float(np.mean(base_f1)),
            'std': float(np.std(base_f1, ddof=1)),
            'median': float(np.median(base_f1)),
            'min': float(np.min(base_f1)),
            'max': float(np.max(base_f1))
        },
        'custom_f1': {
            'mean': float(np.mean(custom_f1)),
            'std': float(np.std(custom_f1, ddof=1)),
            'median': float(np.median(custom_f1)),
            'min': float(np.min(custom_f1)),
            'max': float(np.max(custom_f1))
        },
        'improvement': {
            'mean': float(np.mean(improvements)),
            'std': float(np.std(improvements, ddof=1)),
            'median': float(np.median(improvements)),
            'min': float(np.min(improvements)),
            'max': float(np.max(improvements))
        }
    }
    
    # Test normality
    print(f"\nNormality Tests:")
    base_normal, base_p, base_msg = test_normality(base_f1, "Base F1")
    custom_normal, custom_p, custom_msg = test_normality(custom_f1, "Custom F1")
    improvement_normal, improvement_p, improvement_msg = test_normality(improvements, "Improvements")
    
    print(f"  {base_msg}")
    print(f"  {custom_msg}")  
    print(f"  {improvement_msg}")
    
    results['normality_tests'] = {
        'base_f1': {'is_normal': base_normal, 'p_value': float(base_p)},
        'custom_f1': {'is_normal': custom_normal, 'p_value': float(custom_p)},
        'improvement': {'is_normal': improvement_normal, 'p_value': float(improvement_p)}
    }
    
    # Paired t-test (parametric)
    print(f"\nPaired T-Test (Customized vs Base):")
    t_stat, t_p = ttest_rel(custom_f1, base_f1)
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {t_p:.4f}")
    print(f"  Significant: {'Yes' if t_p < 0.05 else 'No'}")
    
    # Wilcoxon signed-rank test (non-parametric alternative)
    print(f"\nWilcoxon Signed-Rank Test (Customized vs Base):")
    try:
        w_stat, w_p = wilcoxon(custom_f1, base_f1)
        print(f"  W-statistic: {w_stat:.4f}")
        print(f"  p-value: {w_p:.4f}")
        print(f"  Significant: {'Yes' if w_p < 0.05 else 'No'}")
    except ValueError as e:
        print(f"  Could not perform Wilcoxon test: {e}")
        w_stat, w_p = None, None
    
    # Effect size (Cohen's d)
    cohens_d = calculate_cohens_d(custom_f1, base_f1)
    print(f"\nEffect Size:")
    print(f"  Cohen's d: {cohens_d:.4f}")
    
    # Interpret effect size
    if abs(cohens_d) < 0.2:
        effect_interpretation = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_interpretation = "small"
    elif abs(cohens_d) < 0.8:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"
    
    print(f"  Interpretation: {effect_interpretation} effect")
    
    # Bootstrap confidence intervals
    print(f"\nBootstrap 95% Confidence Intervals:")
    improvement_ci = bootstrap_confidence_interval(improvements, np.mean)
    print(f"  Mean improvement: [{improvement_ci[0]:.4f}, {improvement_ci[1]:.4f}]")
    
    # Count positive/negative improvements
    positive_improvements = np.sum(improvements > 0)
    negative_improvements = np.sum(improvements < 0) 
    zero_improvements = np.sum(improvements == 0)
    
    print(f"\nImprovement Patterns:")
    print(f"  Positive improvements: {positive_improvements}/{len(improvements)} ({positive_improvements/len(improvements)*100:.1f}%)")
    print(f"  Negative improvements: {negative_improvements}/{len(improvements)} ({negative_improvements/len(improvements)*100:.1f}%)")
    print(f"  Zero improvements: {zero_improvements}/{len(improvements)} ({zero_improvements/len(improvements)*100:.1f}%)")
    
    # Sign test
    print(f"\nSign Test (H0: median improvement = 0):")
    n_positive = positive_improvements
    n_total = len(improvements[improvements != 0])  # Exclude ties
    if n_total > 0:
        sign_p = 2 * min(stats.binom.cdf(n_positive, n_total, 0.5), 1 - stats.binom.cdf(n_positive-1, n_total, 0.5))
        print(f"  Positive/Total: {n_positive}/{n_total}")
        print(f"  p-value: {sign_p:.4f}")
        print(f"  Significant: {'Yes' if sign_p < 0.05 else 'No'}")
    else:
        sign_p = None
        print(f"  Cannot perform sign test (no non-zero improvements)")
    
    results['statistical_tests'] = {
        'paired_ttest': {
            't_statistic': float(t_stat),
            'p_value': float(t_p),
            'significant': bool(t_p < 0.05)
        },
        'wilcoxon_test': {
            'w_statistic': float(w_stat) if w_stat is not None else None,
            'p_value': float(w_p) if w_p is not None else None,
            'significant': bool(w_p < 0.05) if w_p is not None else None
        },
        'effect_size': {
            'cohens_d': float(cohens_d),
            'interpretation': effect_interpretation
        },
        'confidence_intervals': {
            'mean_improvement_95ci': [float(improvement_ci[0]), float(improvement_ci[1])]
        },
        'improvement_patterns': {
            'positive_count': int(positive_improvements),
            'negative_count': int(negative_improvements),
            'zero_count': int(zero_improvements),
            'positive_percentage': float(positive_improvements/len(improvements)*100)
        },
        'sign_test': {
            'n_positive': int(n_positive) if sign_p is not None else None,
            'n_total': int(n_total) if sign_p is not None else None,
            'p_value': float(sign_p) if sign_p is not None else None,
            'significant': bool(sign_p < 0.05) if sign_p is not None else None
        }
    }
    
    return results

def generate_participant_analysis(df):
    """Generate detailed per-participant analysis."""
    participant_analysis = []
    
    for _, row in df.iterrows():
        analysis = {
            'participant': row['participant'],
            'base_f1': row['base_f1'],
            'custom_f1': row['custom_f1'],
            'f1_improvement': row['f1_improvement'],
            'f1_improvement_percent': row['f1_improvement_percent'],
            'base_train_samples': row['base_train_samples'],
            'target_train_samples': row['target_train_samples'],
            'target_test_samples': row['target_test_samples'],
            'base_epochs': row['base_epochs'],
            'custom_epochs': row['custom_epochs'],
            'improvement_category': 'positive' if row['f1_improvement'] > 0 else 'negative' if row['f1_improvement'] < 0 else 'none'
        }
        participant_analysis.append(analysis)
    
    return pd.DataFrame(participant_analysis)

def create_summary_report(results, participant_df, output_file):
    """Create human-readable summary report."""
    report = f"""
LOPO Customization Experiment - Statistical Analysis Report
==========================================================

Generated: {results['timestamp']}
Participants: {results['n_participants']} ({', '.join(results['participants'])})

MAIN FINDINGS
=============

Performance Overview:
  • Base Model F1:       {results['descriptive_stats']['base_f1']['mean']:.4f} ± {results['descriptive_stats']['base_f1']['std']:.4f}
  • Customized Model F1: {results['descriptive_stats']['custom_f1']['mean']:.4f} ± {results['descriptive_stats']['custom_f1']['std']:.4f}
  • Mean Improvement:    {results['descriptive_stats']['improvement']['mean']:.4f} ± {results['descriptive_stats']['improvement']['std']:.4f}
  
Statistical Significance:
  • Paired t-test p-value:        {results['statistical_tests']['paired_ttest']['p_value']:.4f} ({'Significant' if results['statistical_tests']['paired_ttest']['significant'] else 'Not significant'})
  • Wilcoxon test p-value:        {results['statistical_tests']['wilcoxon_test']['p_value']:.4f if results['statistical_tests']['wilcoxon_test']['p_value'] else 'N/A'} ({'Significant' if results['statistical_tests']['wilcoxon_test']['significant'] else 'Not significant' if results['statistical_tests']['wilcoxon_test']['significant'] is not None else 'N/A'})
  • Effect size (Cohen's d):      {results['statistical_tests']['effect_size']['cohens_d']:.4f} ({results['statistical_tests']['effect_size']['interpretation']} effect)

Improvement Patterns:
  • Participants with positive improvement: {results['statistical_tests']['improvement_patterns']['positive_count']}/{results['n_participants']} ({results['statistical_tests']['improvement_patterns']['positive_percentage']:.1f}%)
  • Participants with negative improvement: {results['statistical_tests']['improvement_patterns']['negative_count']}/{results['n_participants']}
  • 95% CI for mean improvement: [{results['statistical_tests']['confidence_intervals']['mean_improvement_95ci'][0]:.4f}, {results['statistical_tests']['confidence_intervals']['mean_improvement_95ci'][1]:.4f}]

DETAILED RESULTS
================

Per-Participant Performance:
"""
    
    for _, row in participant_df.iterrows():
        report += f"  • {row['participant']}: {row['base_f1']:.4f} → {row['custom_f1']:.4f} ({row['f1_improvement']:+.4f}, {row['improvement_category']})\n"
    
    report += f"""
INTERPRETATION
==============

Statistical Conclusion:
The {'paired t-test shows significant improvement' if results['statistical_tests']['paired_ttest']['significant'] else 'paired t-test shows no significant improvement'} 
(p = {results['statistical_tests']['paired_ttest']['p_value']:.4f}) with a {results['statistical_tests']['effect_size']['interpretation']} effect size 
(Cohen's d = {results['statistical_tests']['effect_size']['cohens_d']:.4f}).

Practical Significance:
{results['statistical_tests']['improvement_patterns']['positive_percentage']:.0f}% of participants showed positive improvement from customization.
The mean improvement of {results['descriptive_stats']['improvement']['mean']:.4f} F1 points represents a 
{results['descriptive_stats']['improvement']['mean']/results['descriptive_stats']['base_f1']['mean']*100:.1f}% relative improvement over the base model.

RESEARCH IMPLICATIONS
====================

Publication Readiness:
• {'✅' if results['statistical_tests']['paired_ttest']['significant'] else '❌'} Statistical significance achieved (p < 0.05)
• {'✅' if results['statistical_tests']['effect_size']['cohens_d'] > 0.5 else '❌'} Meaningful effect size (Cohen's d > 0.5)  
• {'✅' if results['statistical_tests']['improvement_patterns']['positive_percentage'] > 70 else '❌'} Majority of participants benefit (>70% positive)
• {'✅' if results['n_participants'] >= 5 else '❌'} Adequate sample size (n ≥ 5)

Next Steps:
1. {'Generate publication figures' if results['statistical_tests']['paired_ttest']['significant'] else 'Consider additional data collection or methodology refinement'}
2. {'Write results section with these statistics' if results['statistical_tests']['paired_ttest']['significant'] else 'Analyze why customization did not show significant improvement'}  
3. {'Prepare for peer review' if results['statistical_tests']['paired_ttest']['significant'] else 'Refine experimental design'}

TECHNICAL DETAILS
=================

Normality Tests (Shapiro-Wilk):
  • Base F1 scores:    p = {results['normality_tests']['base_f1']['p_value']:.4f} ({'Normal' if results['normality_tests']['base_f1']['is_normal'] else 'Non-normal'})
  • Custom F1 scores:  p = {results['normality_tests']['custom_f1']['p_value']:.4f} ({'Normal' if results['normality_tests']['custom_f1']['is_normal'] else 'Non-normal'})
  • Improvements:      p = {results['normality_tests']['improvement']['p_value']:.4f} ({'Normal' if results['normality_tests']['improvement']['is_normal'] else 'Non-normal'})

Recommendation: {'Use parametric tests (t-test)' if all([results['normality_tests']['base_f1']['is_normal'], results['normality_tests']['custom_f1']['is_normal']]) else 'Consider non-parametric tests (Wilcoxon)'} for primary analysis.
"""
    
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"📋 Summary report saved: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze LOPO experiment results')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing LOPO results (with lopo_summary.csv)')
    parser.add_argument('--output_dir', type=str, 
                       help='Output directory for analysis results (default: same as results_dir)')
    parser.add_argument('--alpha', type=float, default=0.05,
                       help='Significance level for statistical tests (default: 0.05)')
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.results_dir):
        print(f"❌ Error: Results directory {args.results_dir} does not exist")
        sys.exit(1)
    
    output_dir = args.output_dir if args.output_dir else args.results_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🔬 STATISTICAL ANALYSIS OF LOPO EXPERIMENTS")
    print(f"📂 Results directory: {args.results_dir}")
    print(f"📊 Output directory: {output_dir}")
    print(f"🎯 Significance level: α = {args.alpha}")
    
    try:
        # Load results
        print(f"\n📥 Loading LOPO results...")
        df = load_lopo_results(args.results_dir)
        
        # Perform statistical analysis
        print(f"\n🧮 Performing statistical analysis...")
        results = perform_statistical_tests(df)
        
        # Generate per-participant analysis
        print(f"\n👥 Generating participant analysis...")
        participant_df = generate_participant_analysis(df)
        
        # Save results
        print(f"\n💾 Saving analysis results...")
        
        # JSON results
        json_file = os.path.join(output_dir, 'statistical_analysis.json')
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"📄 Detailed results: {json_file}")
        
        # Participant CSV
        csv_file = os.path.join(output_dir, 'participant_analysis.csv')
        participant_df.to_csv(csv_file, index=False)
        print(f"📈 Participant breakdown: {csv_file}")
        
        # Summary report
        report_file = os.path.join(output_dir, 'statistical_summary.txt')
        create_summary_report(results, participant_df, report_file)
        
        # Final summary
        print(f"\n📊 ANALYSIS COMPLETE")
        print(f"✅ Statistical significance: {'YES' if results['statistical_tests']['paired_ttest']['significant'] else 'NO'}")
        print(f"📈 Mean improvement: {results['descriptive_stats']['improvement']['mean']:+.4f} F1 points")
        print(f"👥 Participants with positive improvement: {results['statistical_tests']['improvement_patterns']['positive_count']}/{results['n_participants']}")
        print(f"🎯 Effect size: {results['statistical_tests']['effect_size']['cohens_d']:.4f} ({results['statistical_tests']['effect_size']['interpretation']})")
        
        if results['statistical_tests']['paired_ttest']['significant']:
            print(f"\n🎉 Results are publication-ready!")
            print(f"   Next: Generate figures and write manuscript")
        else:
            print(f"\n⚠️  No significant improvement found")
            print(f"   Consider: Additional data, different methodology, or refined analysis")
            
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()