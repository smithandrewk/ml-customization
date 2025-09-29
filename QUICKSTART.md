# Quick Reference: After Running Experiments

## 🚀 One-Line Solution

After your training completes, just run:

```bash
./post_experiment_evaluation.sh --all
```

**This automatically:**
1. ✓ Computes test set F1 scores for all experiments
2. ✓ Generates manuscript results (results.tex, abstract_stats.tex)
3. ✓ Creates all figures with test set data
4. ✓ Produces a summary report

## 📋 What This Fixes

### ❌ Before (Using Validation Metrics - WRONG):
```python
# figure2.py, figure3.py, figure4.py
base_f1 = metrics['best_target_val_f1_from_best_base_model']  # BIASED
target_f1 = metrics['best_target_val_f1']                      # BIASED
```

### ✅ After (Using Test Metrics - CORRECT):
```python
# All figure scripts now use:
base_f1 = metrics['base_test_f1']    # UNBIASED - held-out test set
target_f1 = metrics['test_f1']       # UNBIASED - held-out test set
```

## 📊 Expected Output

The script will:

1. **Evaluate test sets** (2-3 minutes)
   ```
   ✓ Base model test F1: 0.8672
   ✓ Personalized model test F1: 0.8622
   ✓ Improvement: -0.0050 (-0.6%)
   ```

2. **Generate manuscript files**
   - `manuscript/results.tex` - Tables with test set performance
   - `manuscript/abstract_stats.tex` - Stats for abstract

3. **Create figures**
   - `figures/figure2.pdf` - Main results
   - `figures/figure3.pdf` - Training dynamics
   - `figures/figure4.pdf` - Data efficiency

4. **Save summary report**
   - `evaluation_report_YYYYMMDD_HHMMSS.txt`

## 🔧 Advanced Usage

### Evaluate specific experiment only:
```bash
./post_experiment_evaluation.sh my_experiment_name
```

### Use different GPU:
```bash
./post_experiment_evaluation.sh --all --device cuda:1
```

### Manual evaluation (if script fails):
```bash
# Step 1: Compute test metrics
python3 evaluate_on_test_set.py --all --device cuda:0

# Step 2: Generate results
cd manuscript
python3 generate_results.py my_experiment_name

# Step 3: Generate figures
python3 figure2.py --experiment my_experiment_name
python3 figure3.py --experiment my_experiment_name
python3 figure4.py
cd ..

# Step 4: Copy figures
cd figures
cp figure2_my_experiment_name.pdf figure2.pdf
cp figure3_my_experiment_name.pdf figure3.pdf
```

## 📁 What Gets Updated

```
experiments/
└── my_experiment/
    └── fold0_participant/
        └── metrics.json        # ← NEW: base_test_f1, test_f1 added

manuscript/
├── results.tex                 # ← UPDATED: test set metrics
├── abstract_stats.tex          # ← UPDATED: test set metrics
└── manuscript.pdf              # ← Recompile after running script

figures/
├── figure2.pdf                 # ← UPDATED: test set metrics
├── figure3.pdf                 # ← UPDATED: test set metrics
└── figure4.pdf                 # ← UPDATED: test set metrics
```

## ⚠️ Important Notes

### Always Run This Script When:
- ✓ New experiments complete
- ✓ Re-running experiments
- ✓ Before compiling manuscript
- ✓ Before generating any figures

### Never Use:
- ❌ Validation metrics in manuscript
- ❌ Figures generated before running this script
- ❌ Results tables without test set evaluation

## 🎯 For Full n=15 Experiments

**Before training:**
1. Edit `train.py` line 38 to include all 15 participants
2. Run training for all 15 folds

**After training:**
```bash
./post_experiment_evaluation.sh --all
```

Done! All results will use test set metrics.

## 💡 Why This Matters

**Validation Set Results (n=3):**
- Responder rate: 100% (3/3) ← Optimistic!
- Mean improvement: +9.8%    ← Inflated!

**Test Set Results (n=3):**
- Responder rate: 33% (1/3)  ← Realistic
- Mean improvement: +5.7%    ← Honest

**Difference:** Validation metrics showed overfitting for 2/3 participants!

---

**See `WORKFLOW.md` for detailed documentation.**