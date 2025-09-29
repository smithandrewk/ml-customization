# Experiment Workflow: Train → Evaluate → Analyze

This document describes the complete workflow for running experiments, computing test set metrics, and generating manuscript results.

## Quick Start

After running training experiments, simply execute:

```bash
./post_experiment_evaluation.sh --all
```

This will:
1. ✓ Compute test set metrics for all experiments
2. ✓ Generate manuscript results (results.tex, abstract_stats.tex)
3. ✓ Generate all figures (figure2.pdf, figure3.pdf, figure4.pdf)
4. ✓ Create a summary report

## Complete Workflow

### Step 1: Run Training Experiments

```bash
# Example: Full fine-tuning with 100% target data
EXPERIMENT_NAME="my_experiment_$(date +%Y%m%d_%H%M%S)"

# Run for each fold (participant)
for fold in 0 1 2; do
    python3 train.py \
        --fold $fold \
        --device 0 \
        --batch_size 128 \
        --model test \
        --use_augmentation \
        --prefix $EXPERIMENT_NAME \
        --early_stopping_patience 50 \
        --early_stopping_patience_target 50 \
        --mode full_fine_tuning \
        --target_data_pct 1.0
done
```

This creates: `experiments/$EXPERIMENT_NAME/fold{0,1,2}_<participant>/`

Each fold directory contains:
- `hyperparameters.json` - Training configuration
- `metrics.json` - Validation metrics (will be updated with test metrics)
- `losses.json` - Training/validation curves
- `best_base_model.pt` - Best base model checkpoint
- `best_target_model.pt` - Best personalized model checkpoint

### Step 2: Compute Test Set Metrics

**Option A: Evaluate all experiments**
```bash
./post_experiment_evaluation.sh --all
```

**Option B: Evaluate specific experiment**
```bash
./post_experiment_evaluation.sh my_experiment_20250929_112052
```

**Option C: Manual evaluation**
```bash
python3 evaluate_on_test_set.py --experiment my_experiment_20250929_112052 --device cuda:0
```

This adds to each `metrics.json`:
```json
{
    "base_test_f1": 0.867,      // Base model on held-out test set
    "test_f1": 0.862,            // Personalized model on held-out test set
    "base_test_loss": 0.015,
    "test_loss": 0.016
}
```

### Step 3: Generate Manuscript Results

The post-experiment script automatically generates:

**Results tables:**
```bash
cd manuscript
python3 generate_results.py my_experiment_20250929_112052
```

Creates:
- `manuscript/results.tex` - Performance tables with test set metrics
- `manuscript/abstract_stats.tex` - LaTeX macros for abstract

**Figures:**
```bash
# Figure 2: Main results (performance improvements)
python3 manuscript/figure2.py --experiment my_experiment_20250929_112052

# Figure 3: Training dynamics
python3 manuscript/figure3.py --experiment my_experiment_20250929_112052

# Figure 4: Data efficiency analysis (uses all experiments)
python3 manuscript/figure4.py
```

### Step 4: Compile Manuscript

```bash
cd manuscript
pdflatex manuscript.tex
pdflatex manuscript.tex  # Run twice for references
```

Or use the Makefile:
```bash
cd manuscript
make
```

## Understanding the Data Splits

### Three-Way Split (Train/Val/Test)

Each participant's data is split into:
- **Training (60%)**: Used for model optimization
- **Validation (20%)**: Used for early stopping and hyperparameter selection
- **Test (20%)**: Held-out for final unbiased evaluation

```
Participant Data
    ├── Train (60%)    → Model training
    ├── Val (20%)      → Early stopping (used during training)
    └── Test (20%)     → Final evaluation (NEVER seen during training)
```

### Why Test Set Metrics Matter

**Validation metrics are BIASED:**
- Used for early stopping → model is optimized to perform well on validation set
- Can overfit to validation set
- Not a true measure of generalization

**Test metrics are UNBIASED:**
- Never seen during training or model selection
- True measure of generalization to new data
- Required for publication-quality results

**Example from our n=3 results:**
```
              Validation    Test      Difference
asfik:        +1.7%        -0.6%     -2.3% (overfitting!)
ejaz:         +1.4%        -1.7%     -3.1% (overfitting!)
tonmoy:       +32.9%       +23.2%    -9.7% (still good, but inflated)
```

## Running Full n=15 Experiments

### Update Participant List

Edit `train.py` line 38:
```python
# From (n=3 for testing):
'participants': ['tonmoy','asfik','ejaz'],

# To (n=15 full dataset):
'participants': [
    'tonmoy', 'alsaad', 'anam', 'asfik', 'ejaz',
    'iftakhar', 'unk1', 'dennis', 'participant9',
    'participant10', 'participant11', 'participant12',
    'participant13', 'participant14', 'participant15'
],
```

### Run All Folds

```bash
EXPERIMENT_NAME="full_n15_$(date +%Y%m%d_%H%M%S)"

for fold in {0..14}; do
    echo "Training fold $fold..."
    python3 train.py \
        --fold $fold \
        --device 0 \
        --batch_size 128 \
        --model test \
        --use_augmentation \
        --prefix $EXPERIMENT_NAME \
        --early_stopping_patience 50 \
        --early_stopping_patience_target 50 \
        --mode full_fine_tuning \
        --target_data_pct 1.0 \
        > logs/fold${fold}.log 2>&1 &

    # Optional: stagger jobs to avoid GPU memory issues
    sleep 60
done

# Wait for all jobs to complete
wait

# Evaluate test sets
./post_experiment_evaluation.sh --all
```

### Parallel Execution (if multiple GPUs available)

```bash
# GPU 0: folds 0-4
for fold in {0..4}; do
    python3 train.py --fold $fold --device 0 --prefix $EXPERIMENT_NAME ... &
done

# GPU 1: folds 5-9
for fold in {5..9}; do
    python3 train.py --fold $fold --device 1 --prefix $EXPERIMENT_NAME ... &
done

# GPU 2: folds 10-14
for fold in {10..14}; do
    python3 train.py --fold $fold --device 2 --prefix $EXPERIMENT_NAME ... &
done

wait
./post_experiment_evaluation.sh --all
```

## File Organization

```
ml-customization/
├── train.py                          # Main training script
├── evaluate_on_test_set.py          # Compute test metrics
├── post_experiment_evaluation.sh    # Complete evaluation pipeline
├── experiments/
│   └── my_experiment_20250929/
│       ├── fold0_participant1/
│       │   ├── hyperparameters.json
│       │   ├── metrics.json          # ✓ Contains test_f1, base_test_f1
│       │   ├── losses.json
│       │   ├── best_base_model.pt
│       │   └── best_target_model.pt
│       ├── fold1_participant2/
│       └── ...
├── manuscript/
│   ├── manuscript.tex                # Main manuscript
│   ├── results.tex                   # ✓ Auto-generated from test metrics
│   ├── abstract_stats.tex            # ✓ Auto-generated stats
│   ├── generate_results.py
│   ├── figure2.py                    # ✓ Uses test metrics
│   ├── figure3.py                    # ✓ Uses test metrics
│   └── figure4.py                    # ✓ Uses test metrics
└── figures/
    ├── figure2.pdf                   # ✓ Generated from test metrics
    ├── figure3.pdf                   # ✓ Generated from test metrics
    └── figure4.pdf                   # ✓ Generated from test metrics
```

## Troubleshooting

### Test metrics not computed

**Problem:** `metrics.json` doesn't have `test_f1` field

**Solution:**
```bash
python3 evaluate_on_test_set.py --experiment <name> --device cuda:0
```

### Figures show old validation metrics

**Problem:** Figures generated before test evaluation

**Solution:**
```bash
# Re-evaluate test sets
./post_experiment_evaluation.sh --all

# Or manually regenerate figures
python3 manuscript/figure2.py --experiment <name>
python3 manuscript/figure3.py --experiment <name>
python3 manuscript/figure4.py
```

### Missing model checkpoint files

**Problem:** `best_target_model.pt` not found

**Cause:** Training didn't reach target phase, or crashed during training

**Solution:** Re-run training for that fold

### GPU out of memory

**Problem:** Evaluation crashes with CUDA OOM

**Solution:**
```bash
# Use CPU for evaluation
python3 evaluate_on_test_set.py --all --device cpu

# Or reduce batch size in evaluate_on_test_set.py (line ~55)
# Change: batch_size = hyperparams.get('batch_size', 128)
# To:     batch_size = 32
```

## Best Practices

### ✅ DO:
- Always run `post_experiment_evaluation.sh` after training completes
- Check that `test_f1` exists in metrics.json before generating figures
- Use test set metrics for all manuscript results
- Document your experiment names clearly
- Keep logs of training runs

### ❌ DON'T:
- Don't use validation metrics for manuscript results
- Don't regenerate figures without re-evaluating test sets
- Don't modify test sets after seeing results
- Don't skip test evaluation even for "quick experiments"

## Key Takeaways

1. **Always evaluate test sets after training** using `post_experiment_evaluation.sh`
2. **All manuscript results use test set metrics** (figures, tables, abstract)
3. **Validation metrics are biased** - only use for early stopping during training
4. **Test metrics are unbiased** - use for all publication results
5. **The workflow is automated** - one script does everything after training completes

---

Questions? Check `evaluate_on_test_set.py --help` or `post_experiment_evaluation.sh --help`