# Two-Phase Distributed Training - Summary

## What We Built

A sophisticated distributed training system that eliminates redundant base model training through intelligent caching and two-phase execution.

## The Problem You Had

When running experiments like:
- `full_fine_tuning` with `target_batch_pct=0.25`
- `full_fine_tuning` with `target_batch_pct=0.50`
- `target_only_fine_tuning`

All experiments using the same base participants were training the **same base model multiple times** - wasting significant compute time.

## The Solution

### Two-Phase Architecture

**Phase 1: Train Base Models (Once)**
1. Extract all unique base model configurations from job list
2. Train each unique base model in parallel (distributed)
3. Copy all base models back to master machine
4. Cache in `experiments/base_models/{hash}/`

**Phase 2: Run Experiments (Reuse Base Models)**
1. For each experiment job:
   - Check if it needs a base model
   - Copy base model from master to worker (if not already cached)
   - Run experiment (loads cached base model)
   - Copy results back to master
2. Workers cache base models locally for reuse across jobs

### Key Innovation: Smart Base Model Syncing

- **Master holds all base models** after Phase 1
- **Workers cache base models** locally as needed
- **No waiting or locking** - workers pull base models on-demand
- **Different base models can be used** - jobs aren't blocked waiting for one base model

## Files Created

### Core System
1. **`oct1_train.py`** (modified)
   - Added `--target_batch_pct` for batch composition control
   - Added `base_only` mode for Phase 1 training
   - Added `get_base_model_hash()` for identifying base models
   - Added `load_or_train_base_model()` for caching logic

2. **`distributed_train_helper.py`** (new)
   - `extract_unique_base_models()` - finds unique base configs
   - `generate_base_model_training_jobs()` - creates Phase 1 jobs
   - `get_base_model_hash_from_job()` - computes base model hash
   - Analysis tool: `python distributed_train_helper.py jobs.json`

3. **`run_two_phase_training.py`** (new)
   - Master orchestrator for two-phase execution
   - Runs Phase 1, then Phase 2 automatically
   - Tracks base model cache, skips training if exists

4. **`run_distributed_training.py`** (modified)
   - Added `copy_base_model_to_remote()` - sync base models to workers
   - Added `copy_base_model_from_remote()` - copy trained base models back
   - Added remote cache checking to avoid duplicate copies
   - Handles `base_only` mode jobs

### Documentation & Examples
5. **`TWO_PHASE_TRAINING.md`** - Complete usage guide
6. **`example_two_phase_workflow.sh`** - Runnable example
7. **`example_batch_composition.py`** - Batch composition search example
8. **`SUMMARY_TWO_PHASE.md`** - This file

## Usage

### Simple Workflow

```bash
# 1. Generate jobs
python example_batch_composition.py

# 2. (Optional) Analyze base models needed
python distributed_train_helper.py jobs_batch_composition.json

# 3. Run two-phase training
python run_two_phase_training.py \
    --cluster-config cluster_config.json \
    --jobs-config jobs_batch_composition.json \
    --script-path oct1_train.py
```

### What Happens

**Without two-phase training:**
- 15 experiments × 2 phases = 30 training runs total
- Lots of redundant base model training

**With two-phase training:**
- Phase 1: 1 base model training job
- Phase 2: 15 target fine-tuning jobs
- **Savings: ~14 redundant base trainings eliminated!**

## Features

### Batch Composition Control (`--target_batch_pct`)

New hyperparameter for fine-tuning phase:
- Controls percentage of each batch from target participant
- Example: `--target_batch_pct 0.25` = 25% target, 75% base per batch
- Allows investigating: "Does batch composition affect personalization?"

### Base Model Caching

- **Automatic deduplication**: Same base config = same hash = reused
- **Hash includes**: participants, model, batch_size, lr, augmentation
- **Hash excludes**: fold, target_data_pct, target_batch_pct, prefix
- **Persistent**: Base models saved permanently in `experiments/base_models/`

### Distributed Syncing

- **Phase 1**: All base models copied to master
- **Phase 2**: Base models synced to workers on-demand
- **Worker caching**: Workers keep base models for future jobs
- **No coordination**: No locks, no waiting, pure pull-based

## Example Use Cases

### 1. Batch Composition Search
Search over `target_batch_pct = [0.1, 0.25, 0.5, 0.75, 0.9]`
- All experiments share same base model
- Only 1 base model trained
- 5× fewer base training runs!

### 2. Target Data Amount Search
Search over `target_data_pct = [0.05, 0.1, 0.25, 0.5, 1.0]`
- All experiments share same base model
- Only 1 base model trained
- 5× fewer base training runs!

### 3. Mixed Hyperparameter Search
Vary both `target_batch_pct` AND `target_data_pct`
- 5 × 5 = 25 experiments
- All share same base model
- Only 1 base model trained
- 25× fewer base training runs!

## Performance Benefits

### Compute Savings
For N experiments sharing a base model:
- **Old**: N complete training runs
- **New**: 1 base training + N target training
- **Savings**: (N-1) base training runs

### Time Savings (Rough Estimates)
Assuming base training = 2 hours, target training = 1 hour:
- **Old**: 15 experiments × 3 hours = 45 GPU-hours
- **New**: 2 hours + (15 × 1 hour) = 17 GPU-hours
- **Savings**: ~60% reduction in total compute time!

### Storage
- Base models: `~100MB` each
- Shared across all experiments with same config
- Workers cache locally (can be cleared when needed)

## Design Decisions

### Why Two Phases Instead of On-Demand?
- **Simpler**: No complex coordination between workers
- **Faster**: Base models trained in parallel upfront
- **Predictable**: Know exactly what Phase 1 and Phase 2 will do
- **Flexible**: Can skip Phase 1 if base models already exist

### Why Hash-Based Caching?
- **Deterministic**: Same config always produces same hash
- **Automatic**: No manual base model management
- **Safe**: Different configs never collide

### Why Master-Worker Sync?
- **Single source of truth**: Master has all base models after Phase 1
- **On-demand**: Workers only copy what they need
- **Cacheable**: Workers keep base models for future jobs
- **No blocking**: Different workers can use different base models

## Future Enhancements

Possible improvements:
1. **Base model warm pool**: Pre-sync popular base models to all workers
2. **Peer-to-peer sync**: Workers share base models with each other
3. **Incremental base models**: Resume base training from checkpoints
4. **Base model versioning**: Track base model training history

## Testing

To test the system:

```bash
# 1. Run example workflow (dry run)
./example_two_phase_workflow.sh

# 2. Test with small job set
# Edit jobs_config.json to have just 2-3 jobs

# 3. Run two-phase training
python run_two_phase_training.py \
    --cluster-config cluster_config.json \
    --jobs-config jobs_config.json

# 4. Verify results
ls experiments/base_models/        # Should have base models
ls experiments/batch_comp_*/       # Should have experiment results
cat phase1_base_models.json        # Phase 1 log
cat phase2_experiments.json        # Phase 2 log
```

## Troubleshooting

See `TWO_PHASE_TRAINING.md` for detailed troubleshooting guide.

Common issues:
- **Base model not found**: Check Phase 1 logs
- **Hash mismatch**: Ensure configs are identical
- **Disk space**: Clean worker caches periodically
- **Copy failures**: Check SSH keys and permissions
