# Branch Structure

## Main Branch (`main`)

The main branch contains stable, working code:

- **`train.py`** - Original training script
- **`oct1_train.py`** - Enhanced training script with:
  - `--target_batch_pct` for controlling batch composition during fine-tuning
  - Batch composition control (ComposedBatchDataLoader)
  - Improved plotting with None value handling
- **`run_distributed_training.py`** - Distributed job execution across multiple GPU servers
- **`generate_jobs.py`** - Generate job configurations for experiments

### Usage

```bash
# Single experiment with batch composition control
python oct1_train.py \
    --fold 0 \
    --device 0 \
    --batch_size 64 \
    --mode full_fine_tuning \
    --target_batch_pct 0.5 \
    --prefix my_experiment

# Distributed training
python generate_jobs.py  # Creates jobs_config.json
python run_distributed_training.py \
    --cluster-config cluster_config.json \
    --jobs-config jobs_config.json \
    --script-path oct1_train.py
```

## Feature Branch (`feature/two-phase-training`)

⚠️ **Experimental** - Contains two-phase distributed training system with base model caching.

**Status**: Has coordination issues with hash mismatches and race conditions. Not recommended for production use.

### What it contains:
- `run_two_phase_training_v2.py` - Sequential base model training, then parallel experiments
- `distributed_train_helper.py` - Base model extraction and hash computation
- Documentation: `TWO_PHASE_TRAINING.md`, `SUMMARY_TWO_PHASE.md`

### Why it's in a separate branch:
- Complex coordination logic that wasn't working reliably
- Hash computation mismatches between job generation and execution
- Duplicate base model training despite caching attempts
- Needs significant debugging before production use

### To access:
```bash
git checkout feature/two-phase-training
```

## Recommendations

**For current work**: Use `main` branch
- `oct1_train.py` works well for single experiments
- `run_distributed_training.py` handles parallel job execution reliably
- Just run experiments with different configs - if they share base participants, you can manually train the base model once and distribute it

**For future**: The two-phase approach could be revisited with:
- Simpler locking mechanism
- Pre-trained base models as artifacts
- Better separation of concerns between base and target training
