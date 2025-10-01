# Branch Structure

## Main Branch (`main`)

**Clean, minimal code for single-machine training.**

Contains:
- **`train.py`** - Standard training script
- **`oct1_train.py`** - Enhanced training script with batch composition control
- **`run.py`** - Grid search runner for local experiments
- Basic utility scripts

### Usage

```bash
# Single experiment
python train.py \
    --fold 0 \
    --device 0 \
    --batch_size 64 \
    --mode full_fine_tuning \
    --prefix my_experiment

# Enhanced version with batch composition
python oct1_train.py \
    --fold 0 \
    --device 0 \
    --batch_size 64 \
    --mode full_fine_tuning \
    --target_batch_pct 0.5 \
    --prefix my_experiment
```

## Feature Branch: `feature/distributed-training`

**Stable distributed training across multiple GPU servers.**

Contains main branch code PLUS:
- **`run_distributed_training.py`** - Distribute jobs across GPU cluster
- **`generate_jobs.py`** - Generate job configurations
- **`DISTRIBUTED_TRAINING.md`** - Documentation
- Tmux-based remote execution
- Automatic result copying

### Usage

```bash
git checkout feature/distributed-training

# Generate jobs
python generate_jobs.py  # Creates jobs_config.json

# Run distributed
python run_distributed_training.py \
    --cluster-config cluster_config.json \
    --jobs-config jobs_config.json \
    --script-path train.py
```

### Status
✅ **Stable** - Works reliably for parallel job execution across multiple machines.

## Feature Branch: `feature/two-phase-training`

**Experimental base model caching system.**

Contains distributed training code PLUS:
- `run_two_phase_training_v2.py` - Sequential base model training, then parallel experiments
- `distributed_train_helper.py` - Base model extraction and hash computation
- Base model caching with hash-based deduplication
- Enhanced `oct1_train.py` with `base_only` mode
- Documentation: `TWO_PHASE_TRAINING.md`, `SUMMARY_TWO_PHASE.md`

### Status
⚠️ **Experimental** - Has coordination issues:
- Hash computation mismatches between job generation and execution
- Duplicate base model training despite caching attempts
- Complex coordination logic that wasn't working reliably

### To access:
```bash
git checkout feature/two-phase-training
```

## Branch Hierarchy

```
main (clean, single-machine)
  └── feature/distributed-training (stable, multi-machine)
       └── feature/two-phase-training (experimental, base model caching)
```

## Recommendations

### For current work:
**Use `feature/distributed-training` branch**
- Reliable distributed execution
- No complex coordination issues
- Works across your GPU cluster

### For single-machine experiments:
**Use `main` branch**
- Simpler, cleaner codebase
- No unnecessary distributed training code

### For future work:
The two-phase approach could be revisited with:
- Pre-trained base models as artifacts (manually managed)
- Simpler approach: train base model once, copy to all workers manually
- Better hash computation consistency
