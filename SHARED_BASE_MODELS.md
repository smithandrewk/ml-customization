# Shared Base Models Workflow

**Efficient two-phase training to avoid retraining identical base models**

## Overview

When running large-scale experiments with multiple fine-tuning configurations, many experiments share the same base model (same fold, base participants, hyperparameters). The shared base models workflow trains each unique base model once, then reuses it across multiple fine-tuning experiments.

### Efficiency Gains

Example from a typical grid search:
- **Total experiments**: 1,008
- **Unique base models needed**: 84
- **Efficiency**: Each base model is reused by ~12 fine-tuning jobs
- **Time savings**: Massive reduction in redundant base model training

Jobs sharing the same base model differ only in:
- `mode` (full_fine_tuning vs target_only_fine_tuning)
- `target_data_pct` (0.01, 0.05, 0.125, 0.25, 0.5, 1.0)

## Architecture

### Phase 1: Base Model Training
**Script**: `train_base.py`

Trains base models on base participants only (excluding target participant). Each base model is identified by a hash computed from its configuration:

**Hash includes**:
- fold (determines which participants are base vs target)
- n_base_participants
- model architecture
- data_path, window_size
- batch_size, lr, early_stopping_patience
- augmentation settings (jitter_std, magnitude_range, aug_prob)
- participants list

**Hash excludes** (fine-tuning parameters):
- mode (full_fine_tuning vs target_only_fine_tuning)
- target_data_pct
- early_stopping_patience_target
- prefix (just for experiment organization)

**Outputs**:
- `base_models/{hash}.pt` - Model weights
- `base_models/{hash}_metadata.json` - Training configuration and metrics

### Phase 2: Fine-Tuning
**Script**: `train_finetune.py`

Loads a pre-trained base model and fine-tunes on target participant data.

**Modes**:
- `full_fine_tuning`: Train on base + target data concatenated
- `target_only_fine_tuning`: Train on target data only

**Outputs**:
- `experiments/{prefix}/fold{fold}_{target}/` - Fine-tuned model and results

### Job Generation
**Script**: `generate_jobs.py`

Analyzes grid search parameters and identifies unique base model configurations.

**Outputs**:
- `base_training_jobs.json` - One job per unique base model
- `finetune_jobs.json` - All fine-tuning experiments with `base_model_hash` references

## Usage

### Complete Workflow (Automated)

The easiest way to run the entire two-phase workflow:

```bash
# 1. Generate job configurations
python3 generate_jobs.py

# 2. Run complete two-phase workflow across cluster
python3 run_two_phase_distributed.py \
    --cluster-config cluster_config.json \
    --base-jobs base_training_jobs.json \
    --finetune-jobs finetune_jobs.json
```

This automatically:
1. Trains all base models across cluster
2. Waits for completion
3. Syncs base models to all cluster nodes
4. Runs all fine-tuning jobs in parallel

### Manual Workflow (Step-by-Step)

If you prefer to run each phase manually:

```bash
# 1. Generate job configurations
python3 generate_jobs.py

# 2. Train base models
python3 run_distributed_training.py \
    --cluster-config cluster_config.json \
    --jobs-config base_training_jobs.json \
    --script-path train_base.py

# 3. Sync base models to all nodes
python3 run_two_phase_distributed.py \
    --cluster-config cluster_config.json \
    --skip-base-training \
    --finetune-jobs finetune_jobs.json

# OR manually sync:
# rsync -avz base_models/ user@server:~/ml-customization/base_models/

# 4. Fine-tune on target participants
python3 run_distributed_training.py \
    --cluster-config cluster_config.json \
    --jobs-config finetune_jobs.json \
    --script-path train_finetune.py
```

### Local Testing

Test individual training scripts locally:

```bash
# Train a single base model
python3 train_base.py \
    --fold 0 \
    --device 0 \
    --batch_size 64 \
    --model test \
    --prefix test_base \
    --lr 0.0003 \
    --early_stopping_patience 50 \
    --n_base_participants 5 \
    --data_path data/001_60s_window \
    --window_size 3000 \
    --participants tonmoy asfik alsaad anam ejaz iftakhar unk1 \
    --use_augmentation

# Fine-tune using the base model (replace hash with actual)
python3 train_finetune.py \
    --base_model_hash dd8687760fb0bcb2 \
    --fold 0 \
    --device 0 \
    --batch_size 64 \
    --model test \
    --prefix test_finetune \
    --mode target_only_fine_tuning \
    --target_data_pct 0.1 \
    --lr 0.0003 \
    --early_stopping_patience_target 50 \
    --data_path data/001_60s_window \
    --window_size 3000 \
    --participants tonmoy asfik alsaad anam ejaz iftakhar unk1
```

## Advanced Options

### Skip Base Training
If base models are already trained:

```bash
python3 run_two_phase_distributed.py \
    --cluster-config cluster_config.json \
    --finetune-jobs finetune_jobs.json \
    --skip-base-training
```

### Skip Sync
If base models are already synced to all nodes:

```bash
python3 run_two_phase_distributed.py \
    --cluster-config cluster_config.json \
    --finetune-jobs finetune_jobs.json \
    --skip-base-training \
    --skip-sync
```

### Custom Job Files
Use different job configuration files:

```bash
python3 run_two_phase_distributed.py \
    --cluster-config cluster_config.json \
    --base-jobs my_base_jobs.json \
    --finetune-jobs my_finetune_jobs.json
```

## Files and Directories

```
ml-customization/
├── train_base.py              # Phase 1: Train base models
├── train_finetune.py          # Phase 2: Fine-tune on targets
├── generate_jobs.py           # Generate job configurations
├── run_distributed_training.py  # Single-phase distributed runner
├── run_two_phase_distributed.py # Two-phase orchestrator
├── base_models/               # Created by train_base.py
│   ├── {hash}.pt             # Base model weights
│   └── {hash}_metadata.json  # Training config and metrics
└── experiments/               # Created by train_finetune.py
    └── {prefix}/
        └── fold{N}_{participant}/
            ├── best_model.pt
            ├── metrics.json
            └── losses.json
```

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Job Generation (generate_jobs.py)                        │
│    - Analyze grid search                                     │
│    - Identify unique base models                             │
│    - Output: base_training_jobs.json, finetune_jobs.json    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Base Model Training (train_base.py)                      │
│    - Train base models in parallel across cluster           │
│    - Save to base_models/{hash}.pt                          │
│    - 84 unique base models (example)                        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Sync Base Models                                          │
│    - rsync base_models/ to all cluster nodes                │
│    - Ensures all workers have access to base models         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Fine-Tuning (train_finetune.py)                          │
│    - Load base model from base_models/{hash}.pt             │
│    - Fine-tune on target participant                        │
│    - Run 1,008 jobs in parallel (example)                   │
│    - Save to experiments/{prefix}/fold{N}_{target}/         │
└─────────────────────────────────────────────────────────────┘
```

## Troubleshooting

### Base model not found
**Error**: `Base model not found: base_models/{hash}.pt`

**Solution**: Ensure base training completed successfully before running fine-tuning:
```bash
# Check if base models exist
ls -lh base_models/

# Re-run base training if needed
python3 run_distributed_training.py \
    --cluster-config cluster_config.json \
    --jobs-config base_training_jobs.json \
    --script-path train_base.py
```

### Base model hash mismatch
**Error**: Fine-tuning job references a base model hash that doesn't exist

**Cause**: Job files were generated before base training, and hashing logic changed

**Solution**: Regenerate job files:
```bash
python3 generate_jobs.py
```

### Sync failures
**Error**: rsync fails to some cluster nodes

**Solution**:
1. Verify SSH access to all nodes
2. Check cluster_config.json has correct credentials
3. Manually copy base_models to failed nodes:
```bash
rsync -avz base_models/ user@failed-node:~/ml-customization/base_models/
```

## Benefits

1. **Time Efficiency**: Train each unique base model only once
2. **Resource Efficiency**: No redundant GPU time on identical base training
3. **Consistency**: All experiments with the same base config use identical base models
4. **Flexibility**: Easy to add more fine-tuning experiments without retraining bases
5. **Reproducibility**: Base model hashes ensure exact configuration matching

## Comparison with Original Workflow

### Original (train.py)
- Each experiment trains base model from scratch
- 1,008 total training jobs
- Redundant base model training

### Shared Base Models (train_base.py + train_finetune.py)
- Train each unique base model once: 84 jobs
- Fine-tune on targets: 1,008 jobs
- Total: 1,092 jobs, but ~12x less base training work
- Base training happens first and can be cached indefinitely
