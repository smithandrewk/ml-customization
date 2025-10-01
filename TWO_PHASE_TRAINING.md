# Two-Phase Distributed Training with Base Model Caching

This system eliminates redundant base model training across experiments by using a two-phase approach:

**Phase 1**: Train unique base models (distributed)
**Phase 2**: Run all experiments (distributed, reusing cached base models)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        MASTER MACHINE                        │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Phase 1: Train Base Models (Distributed)               │ │
│  │  • Extract unique base model configs from jobs         │ │
│  │  • Train each base model once (in parallel)            │ │
│  │  • Copy all base models back to master                 │ │
│  │  • Store in experiments/base_models/{hash}/            │ │
│  └────────────────────────────────────────────────────────┘ │
│                            ↓                                 │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Phase 2: Run Experiments (Distributed)                 │ │
│  │  • For each job, check if base model needed            │ │
│  │  • Copy base model from master to worker (if needed)   │ │
│  │  • Worker checks local cache before copying            │ │
│  │  • Run experiment (loads cached base model)            │ │
│  │  • Copy results back to master                         │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            ↓
                   ┌─────────────────┐
                   │  Remote Workers  │
                   │  • GPU Server 1  │
                   │  • GPU Server 2  │
                   │  • GPU Server N  │
                   └─────────────────┘
```

## Quick Start

### 1. Generate experiment jobs

```python
# example: jobs for batch composition search
from distributed_train_helper import *
import json

jobs = []
batch_compositions = [0.1, 0.25, 0.5, 0.75, 0.9]
participants = ['tonmoy', 'asfik', 'ejaz']

for target_batch_pct in batch_compositions:
    for fold in range(len(participants)):
        job = {
            'fold': fold,
            'batch_size': 64,
            'model': 'test',
            'mode': 'full_fine_tuning',
            'target_batch_pct': target_batch_pct,
            'lr': 3e-4,
            'data_path': 'data/001_60s_window',
            'participants': participants,
            'prefix': f'batch_comp_pct{int(target_batch_pct*100)}',
        }
        jobs.append(job)

with open('jobs_config.json', 'w') as f:
    json.dump(jobs, f, indent=2)
```

### 2. Analyze base model requirements (optional)

```bash
python distributed_train_helper.py jobs_config.json
```

This shows:
- How many unique base models are needed
- Which experiments share base models
- Compute savings from base model reuse

### 3. Run two-phase training

```bash
python run_two_phase_training.py \
    --cluster-config cluster_config.json \
    --jobs-config jobs_config.json \
    --script-path oct1_train.py
```

## What Happens

### Phase 1: Base Model Training

1. **Extract unique base models**
   - Analyzes all jobs to find unique base model configurations
   - Base model hash includes: participants, model architecture, batch size, lr, augmentation, etc.
   - Excludes: fold, target_data_pct, target_batch_pct, prefix

2. **Train base models (distributed)**
   - Creates special `mode='base_only'` jobs
   - Workers train base models in parallel
   - Base models copied back to master: `experiments/base_models/{hash}/`

3. **Caching**
   - Each base model identified by 12-character hash
   - Master stores in `experiments/base_models/{hash}/`
   - Contains: `best_base_model.pt`, `base_metrics.json`, `base_losses.json`, `base_config.json`

### Phase 2: Experiment Execution

1. **Job annotation**
   - Each job annotated with `_base_model_hash` field
   - Identifies which base model (if any) the job needs

2. **Base model syncing**
   - Before each job, worker checks if it needs a base model
   - If yes:
     - Check remote cache: does worker already have this base model?
     - If not: copy from master to `~/ml-customization/experiments/base_models/{hash}/`
   - Base models persist on workers for future jobs

3. **Experiment execution**
   - Worker runs `oct1_train.py` with job config
   - Script checks for cached base model (via hash)
   - If found: loads pre-trained base model
   - If not found: impossible (we just copied it!)
   - Runs target fine-tuning phase
   - Results copied back to master

## Example: Batch Composition Search

Searching over 5 batch compositions × 3 folds = 15 experiments:

**Without caching:**
- 15 full training runs (base + target for each)

**With two-phase training:**
- Phase 1: 1 base model training job (all experiments share same base model!)
- Phase 2: 15 target fine-tuning jobs (fast, reusing base model)

**Compute savings:** ~14 base training runs eliminated ✨

## Advanced Options

### Skip Phase 1 (if base models already trained)

```bash
python run_two_phase_training.py \
    --cluster-config cluster_config.json \
    --jobs-config jobs_config.json \
    --skip-phase1
```

### Inspect logs

```bash
# Phase 1 (base model training)
cat phase1_base_models.json | jq

# Phase 2 (experiments)
cat phase2_experiments.json | jq
```

### Check base model cache

```bash
ls -la experiments/base_models/
# Each subdirectory is a base model hash
```

## Files

- `oct1_train.py` - Training script with base model caching
- `run_two_phase_training.py` - Master orchestrator
- `distributed_train_helper.py` - Base model extraction utilities
- `run_distributed_training.py` - Distributed job executor (updated with base model sync)

## Key Features

✅ **Automatic base model deduplication** - train each unique base model once
✅ **Distributed base model training** - Phase 1 runs in parallel
✅ **Smart caching** - workers cache base models locally
✅ **Transparent syncing** - base models copied from master as needed
✅ **No coordination overhead** - no locks, no waiting
✅ **Backward compatible** - works with existing job configs

## Troubleshooting

### Base model not found on worker
- Check `experiments/base_models/` on master - is the base model there?
- Check Phase 1 logs - did base model training succeed?
- Try re-running with `--skip-phase1` to use existing base models

### Base model hash mismatch
- Ensure job config is identical for jobs that should share base models
- Hash includes: participants, model, batch_size, lr, augmentation
- Even small differences create new base models

### Out of disk space on workers
- Base models are deleted from workers after copying to master (Phase 1)
- Base models persist on workers during Phase 2 for caching
- Can manually clean: `ssh worker "rm -rf ~/ml-customization/experiments/base_models/*"`
