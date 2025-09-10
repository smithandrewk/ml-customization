# Ablation Study Guide

This guide explains how to run comprehensive ablation studies for the advanced customization techniques.

## Quick Start

### 1. Run the Experiment Suite
```bash
# Run all experiments for all participants
python run_experiment_suite.py --dataset_dir data/001_tonmoy_60s --participants all

# Run specific experiments for specific participants
python run_experiment_suite.py --dataset_dir data/001_tonmoy_60s --participants tonmoy,asfik,ejaz --experiments baseline,ewc_only,core_trio

# Dry run to see what would be executed
python run_experiment_suite.py --dataset_dir data/001_tonmoy_60s --participants tonmoy --dry_run
```

### 2. Analyze Results
```bash
# Analyze all results with statistical tests
python analyze_ablation_results.py /path/to/ablation_study_results

# Generate CSV and JSON outputs
python analyze_ablation_results.py /path/to/ablation_study_results --output_format both
```

### 3. Generate Figures
```bash
# Generate publication-quality figures
python generate_ablation_figures.py /path/to/ablation_study_results --style publication --format pdf

# Generate presentation figures
python generate_ablation_figures.py /path/to/ablation_study_results --style presentation --format png --dpi 150
```

## Experiment Configurations

### Individual Techniques
- `baseline`: Standard two-phase customization
- `ewc_only`: Elastic Weight Consolidation only
- `layerwise_only`: Layer-wise fine-tuning only  
- `gradual_only`: Gradual unfreezing only
- `augmentation_only`: Data augmentation only
- `coral_only`: CORAL domain adaptation only
- `contrastive_only`: Contrastive learning only
- `ensemble_only`: Ensemble approach only

### Promising Combinations
- `ewc_layerwise`: EWC + Layer-wise fine-tuning
- `ewc_augmentation`: EWC + Data augmentation
- `layerwise_augmentation`: Layer-wise + Data augmentation
- `core_trio`: EWC + Layer-wise + Augmentation
- `domain_adaptation`: CORAL + Contrastive learning
- `all_advanced`: All techniques except ensemble
- `all_plus_ensemble`: All techniques including ensemble

## Output Structure

```
ablation_study_YYYYMMDD_HHMMSS/
├── experiments/           # Individual experiment results
│   ├── tonmoy_baseline/
│   ├── tonmoy_ewc_only/
│   └── ...
├── logs/                 # Execution logs
│   ├── tonmoy_baseline.log
│   └── ...
├── analysis/             # Processed results
│   ├── ablation_results.csv
│   ├── technique_ranking.csv
│   ├── participant_stats.csv
│   └── comprehensive_analysis.json
├── figures/              # Generated visualizations
│   ├── technique_comparison.pdf
│   ├── participant_variation.pdf
│   ├── statistical_significance.pdf
│   ├── category_analysis.pdf
│   ├── technique_impact_matrix.pdf
│   └── summary_dashboard.pdf
├── experiment_metadata.json
└── execution_summary.json
```

## Key Metrics Tracked

### Performance Metrics
- `base_model_test_f1`: F1 score of base model on target participant
- `custom_model_test_f1`: F1 score of customized model on target participant  
- `absolute_improvement`: Absolute F1 improvement (custom - base)
- `percentage_improvement`: Percentage F1 improvement

### Technique Usage
- Boolean flags for each advanced technique used
- Hyperparameter values (lambda values, schedules, etc.)
- Ensemble performance (if applicable)

### Training Details
- Best epochs for each phase
- Sample counts and data splits
- Early stopping metrics used

## Statistical Analysis

The analysis script computes:
- **Descriptive statistics**: Mean, std, median, quartiles, confidence intervals
- **Statistical tests**: Paired/independent t-tests vs baseline
- **Effect sizes**: Cohen's d for practical significance
- **Individual technique impact**: With vs without each technique
- **Participant variation**: How much results vary across participants

## Figure Types Generated

1. **Technique Comparison**: Horizontal bar chart of all techniques ranked by performance
2. **Participant Variation**: Box plots and heatmaps showing individual differences
3. **Statistical Significance**: Significance tests and p-value visualizations
4. **Category Analysis**: Individual vs combination techniques analysis
5. **Technique Impact Matrix**: Impact of each individual technique component
6. **Summary Dashboard**: Comprehensive overview with key statistics

## Best Practices

### For Systematic Studies
1. **Start small**: Run with `--model simple` for faster experiments
2. **Use dry run**: Test your command with `--dry_run` first
3. **Monitor progress**: Check log files during execution
4. **Save metadata**: All configurations and parameters are automatically saved

### For Publication
1. **Multiple runs**: Run each experiment multiple times for robustness
2. **Statistical rigor**: Use appropriate significance levels and effect sizes
3. **Clear reporting**: Include confidence intervals and participant variation
4. **Reproducibility**: Save all configurations and random seeds

### For Presentation
1. **Focus on top techniques**: Use the ranking analysis to highlight best performers
2. **Show significance**: Include statistical significance indicators
3. **Participant generalization**: Show that results hold across participants
4. **Clear categories**: Distinguish individual vs combination techniques

## Troubleshooting

### Common Issues
- **Missing results**: Check log files for errors during training
- **Analysis fails**: Ensure `analyze_ablation_results.py` ran before figure generation
- **Empty figures**: Verify sufficient data points for statistical analysis
- **Memory issues**: Use `--model simple` or reduce `--max_epochs`

### Performance Optimization
- **Parallel execution**: Use `--parallel` flag (experimental)
- **Early stopping**: Ensure proper patience settings in config
- **GPU utilization**: Monitor GPU usage during experiments
- **Storage**: Each experiment generates ~10-50MB of results

## Example Complete Workflow

```bash
# 1. Run comprehensive ablation study
python run_experiment_suite.py \
    --dataset_dir data/001_tonmoy_60s \
    --participants all \
    --model simple \
    --output_dir ablation_results

# 2. Analyze results with statistical tests
python analyze_ablation_results.py ablation_results/ablation_study_20240315_143022/ \
    --output_format both \
    --significance_level 0.05

# 3. Generate publication figures
python generate_ablation_figures.py ablation_results/ablation_study_20240315_143022/ \
    --style publication \
    --format pdf \
    --dpi 300

# 4. Generate presentation figures
python generate_ablation_figures.py ablation_results/ablation_study_20240315_143022/ \
    --style presentation \
    --format png \
    --dpi 150
```

This workflow will give you comprehensive analysis suitable for academic publication with proper statistical testing and professional visualizations.