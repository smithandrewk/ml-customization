# Testing Guide for Shared Base Models System

**How to test the two-phase training system before running on the full cluster**

## Prerequisites

Before testing, ensure you have:
- [ ] Data available at `data/001_60s_window/`
- [ ] Python environment activated: `source env/bin/activate`
- [ ] On branch `feature/shared-base-models`: `git branch` shows `* feature/shared-base-models`

## Quick Test Checklist

### Step 1: Generate Test Jobs (5 min)

```bash
# First, verify you're on the right branch
git branch

# Generate job configurations
python3 generate_jobs.py
```

**Expected output:**
```
Total experiment configurations: 1008
Configs needing base models: 504
Configs without base models (target_only): 0
Unique base models needed: 84

Files created:
  - base_training_jobs.json (84 jobs)
  - finetune_jobs.json (1008 jobs)
```

**Verify:**
```bash
# Check files were created
ls -lh base_training_jobs.json finetune_jobs.json

# Inspect first base job
head -40 base_training_jobs.json

# Inspect first finetune job
head -40 finetune_jobs.json
```

**What to look for:**
- Base jobs should have `_base_model_hash` field
- Fine-tune jobs should have `base_model_hash` field
- `mode: 'full_fine_tuning'` jobs should have base_model_hash != null
- `mode: 'target_only'` jobs (if any) should have base_model_hash = null
- **Important:** `target_only` jobs should NOT duplicate across `n_base_participants`
  - If you have 6 n_base_participants values, full_fine_tuning should have 6× jobs
  - But target_only should only have 1× jobs (n_base_participants is irrelevant)

---

### Step 2: Test Single Base Model Training (10-30 min)

Test training one base model locally before running on cluster.

```bash
# Train a single base model (fold 0, 5 base participants)
python3 train_base.py \
    --fold 0 \
    --device 0 \
    --batch_size 64 \
    --model test \
    --prefix test_base \
    --lr 0.0003 \
    --early_stopping_patience 10 \
    --early_stopping_patience_target 10 \
    --mode full_fine_tuning \
    --target_data_pct 1.0 \
    --n_base_participants 5 \
    --data_path data/001_60s_window \
    --window_size 3000 \
    --participants tonmoy asfik alsaad anam ejaz iftakhar unk1 \
    --use_augmentation \
    --jitter_std 0.005 \
    --magnitude_range 0.98 1.02 \
    --aug_prob 0.3
```

**Expected behavior:**
1. Prints base participants (should exclude fold 0 = tonmoy)
2. Shows base model hash (e.g., `Base model hash: a1b2c3d4e5f6g7h8`)
3. Trains for several epochs
4. Saves to `base_models/{hash}.pt` and `base_models/{hash}_metadata.json`

**Verify:**
```bash
# Check base_models directory was created
ls -lh base_models/

# Should see two files: {hash}.pt and {hash}_metadata.json
# Example: a1b2c3d4e5f6g7h8.pt and a1b2c3d4e5f6g7h8_metadata.json

# Inspect metadata
cat base_models/*.json | jq '.'
```

**Test idempotency (should skip training):**
```bash
# Run the same command again - should skip
python3 train_base.py \
    --fold 0 \
    --device 0 \
    --batch_size 64 \
    --model test \
    --prefix test_base \
    --lr 0.0003 \
    --early_stopping_patience 10 \
    --early_stopping_patience_target 10 \
    --mode full_fine_tuning \
    --target_data_pct 1.0 \
    --n_base_participants 5 \
    --data_path data/001_60s_window \
    --window_size 3000 \
    --participants tonmoy asfik alsaad anam ejaz iftakhar unk1 \
    --use_augmentation \
    --jitter_std 0.005 \
    --magnitude_range 0.98 1.02 \
    --aug_prob 0.3
```

**Expected:** Should print "Base model already exists" and skip training.

---

### Step 3: Test Fine-Tuning (10-30 min)

Test fine-tuning with the base model you just trained.

**Get the base model hash:**
```bash
# Find the hash from the filename
ls base_models/*.pt
# Example output: base_models/a1b2c3d4e5f6g7h8.pt
# Hash is: a1b2c3d4e5f6g7h8
```

**Test target_only_fine_tuning mode:**
```bash
# Replace {HASH} with your actual hash
python3 train_finetune.py \
    --base_model_hash a1b2c3d4e5f6g7h8 \
    --fold 0 \
    --device 0 \
    --batch_size 64 \
    --model test \
    --prefix test_finetune_target_only \
    --mode target_only_fine_tuning \
    --target_data_pct 0.1 \
    --lr 0.0003 \
    --early_stopping_patience 10 \
    --early_stopping_patience_target 10 \
    --n_base_participants 5 \
    --data_path data/001_60s_window \
    --window_size 3000 \
    --participants tonmoy asfik alsaad anam ejaz iftakhar unk1 \
    --use_augmentation \
    --jitter_std 0.005 \
    --magnitude_range 0.98 1.02 \
    --aug_prob 0.3
```

**Expected behavior:**
1. Loads base model from `base_models/{hash}.pt`
2. Prints base model info (participants, metrics)
3. Trains on target data only (tonmoy for fold 0)
4. Saves to `experiments/test_finetune_target_only/fold0_tonmoy/`

**Verify:**
```bash
# Check experiment directory
ls -lh experiments/test_finetune_target_only/fold0_tonmoy/

# Should contain:
# - best_model.pt
# - metrics.json
# - losses.json
# - training_progress.png (if plot was generated)
```

**Test full_fine_tuning mode:**
```bash
python3 train_finetune.py \
    --base_model_hash a1b2c3d4e5f6g7h8 \
    --fold 0 \
    --device 0 \
    --batch_size 64 \
    --model test \
    --prefix test_finetune_full \
    --mode full_fine_tuning \
    --target_data_pct 0.1 \
    --lr 0.0003 \
    --early_stopping_patience 10 \
    --early_stopping_patience_target 10 \
    --n_base_participants 5 \
    --data_path data/001_60s_window \
    --window_size 3000 \
    --participants tonmoy asfik alsaad anam ejaz iftakhar unk1 \
    --use_augmentation \
    --jitter_std 0.005 \
    --magnitude_range 0.98 1.02 \
    --aug_prob 0.3
```

**Expected:** Should train on base + target data combined.

**Test target_only mode (no base model):**
```bash
python3 train_finetune.py \
    --fold 0 \
    --device 0 \
    --batch_size 64 \
    --model test \
    --prefix test_target_only \
    --mode target_only \
    --target_data_pct 0.1 \
    --lr 0.0003 \
    --early_stopping_patience 10 \
    --early_stopping_patience_target 10 \
    --n_base_participants 5 \
    --data_path data/001_60s_window \
    --window_size 3000 \
    --participants tonmoy asfik alsaad anam ejaz iftakhar unk1 \
    --use_augmentation \
    --jitter_std 0.005 \
    --magnitude_range 0.98 1.02 \
    --aug_prob 0.3
```

**Expected:** Should initialize fresh model (no base model loading).

---

### Step 4: Test Job Generation Hash Consistency (5 min)

Verify that hashes in job files match what train_base.py computes.

```bash
# Extract a base job from base_training_jobs.json
python3 -c "
import json
with open('base_training_jobs.json') as f:
    jobs = json.load(f)
    print('First base job:')
    print(json.dumps(jobs[0], indent=2))
"

# Note the _base_model_hash field
# Then run train_base.py with those exact parameters
# The hash it prints should match the _base_model_hash in the JSON
```

**Manual verification:**
1. Take parameters from first base job
2. Run `train_base.py` with those exact parameters
3. Compare printed hash with `_base_model_hash` in JSON
4. **They must match exactly** - if not, there's a hash computation inconsistency

---

### Step 5: Test Distributed System (Small Scale) (30-60 min)

**Prerequisites:**
- [ ] `cluster_config.json` exists and has correct server details
- [ ] SSH access to cluster nodes configured
- [ ] Code synced to cluster nodes

**Create a small test job set:**

Edit `generate_jobs.py` temporarily:
```python
# Change GRID_PARAMS to a small test set
GRID_PARAMS = {
    'batch_size': [64],
    'lr': [3e-4],
    'early_stopping_patience': [10],
    'mode': ['target_only_fine_tuning'],
    'target_data_pct': [0.1],
    'n_base_participants': [2],
}
```

```bash
# Generate small job set
python3 generate_jobs.py

# Should create much fewer jobs
# Example: 7 base jobs, 7 finetune jobs (one per fold)
```

**Run the two-phase workflow:**
```bash
python3 run_two_phase_distributed.py \
    --cluster-config cluster_config.json \
    --base-jobs base_training_jobs.json \
    --finetune-jobs finetune_jobs.json
```

**Expected behavior:**
1. **Phase 1:** Distributes base training jobs across GPUs
2. Waits for all base jobs to complete
3. **Sync:** Copies `base_models/` to all cluster nodes
4. **Phase 2:** Distributes fine-tuning jobs across GPUs
5. Copies results back to local machine

**Monitor progress:**
```bash
# In another terminal, watch the logs
tail -f base_training_log.json
tail -f finetune_training_log.json

# Or SSH to a server and attach to tmux
ssh user@server1 -t 'tmux attach -t ml_training'
```

**Verify results:**
```bash
# Check base models
ls -lh base_models/

# Check experiments
ls -lh experiments/

# Check logs
cat base_training_log.json | jq '.'
cat finetune_training_log.json | jq '.'
```

---

## Common Issues and Solutions

### Issue: "Base model not found"

**Symptoms:** Fine-tuning fails with "Base model not found: base_models/{hash}.pt"

**Causes:**
1. Base model wasn't trained yet
2. Hash mismatch between job generation and training
3. Base models not synced to worker node

**Solutions:**
```bash
# 1. Check if base model exists locally
ls -lh base_models/

# 2. Verify hash in finetune job matches a base model
grep -o '"base_model_hash": "[^"]*"' finetune_jobs.json | sort -u
ls base_models/*.pt

# 3. If distributed, check on worker node
ssh user@worker "ls -lh ~/ml-customization/base_models/"
```

### Issue: Hash mismatch

**Symptoms:** Jobs reference hashes that don't exist

**Cause:** Hash computation differs between `generate_jobs.py` and `train_base.py`

**Solution:**
```bash
# Compare hash functions
diff <(sed -n '/def compute_base_model_hash/,/^def /p' generate_jobs.py) \
     <(sed -n '/def compute_base_model_hash/,/^def /p' train_base.py)

# Should show no differences
```

### Issue: "No base participants available"

**Symptoms:** `train_base.py` fails with "No base participants available for fold X"

**Cause:** Edge case where n_base_participants is too large or only 1 participant total

**Solution:**
```bash
# Check participant count in job
python3 -c "
import json
with open('base_training_jobs.json') as f:
    job = json.load(f)[0]
    print(f'Participants: {job[\"participants\"]}')
    print(f'Fold: {job[\"fold\"]}')
    print(f'n_base: {job[\"n_base_participants\"]}')
"

# Adjust grid params if needed
```

### Issue: Sync fails

**Symptoms:** `rsync` fails to copy base_models to some nodes

**Solutions:**
```bash
# Test SSH connection
ssh user@server "echo Connected"

# Test rsync manually
rsync -avz base_models/ user@server:~/ml-customization/base_models/

# Check SSH key permissions (must be 600)
chmod 600 ~/.ssh/id_rsa

# Check cluster_config.json has correct details
cat cluster_config.json | jq '.'
```

---

## Test Data Cleanup

After testing, clean up generated files:

```bash
# Remove test base models
rm -rf base_models/

# Remove test experiments
rm -rf experiments/test_*/

# Remove job files
rm -f base_training_jobs.json finetune_jobs.json

# Remove logs
rm -f base_training_log.json finetune_training_log.json
```

---

## Production Checklist

Before running the full experiment:

- [ ] Tested single base model training locally
- [ ] Tested fine-tuning with all three modes
- [ ] Verified hash consistency between job generation and training
- [ ] Tested small-scale distributed run successfully
- [ ] Confirmed base model sync works across all cluster nodes
- [ ] Backed up any existing experiments
- [ ] Set correct GRID_PARAMS in `generate_jobs.py`
- [ ] Cluster has sufficient disk space for base_models and experiments
- [ ] All cluster nodes have updated code (`sync_cluster.py`)

---

## Full Production Run

Once all tests pass:

```bash
# 1. Restore original grid params in generate_jobs.py
# 2. Generate full job set
python3 generate_jobs.py

# 3. Review job counts
wc -l base_training_jobs.json finetune_jobs.json

# 4. Estimate runtime
# Example: 84 base jobs @ ~30min each = ~42 GPU-hours
# With 8 GPUs = ~5.25 hours for Phase 1

# 5. Run full workflow
python3 run_two_phase_distributed.py \
    --cluster-config cluster_config.json \
    --base-jobs base_training_jobs.json \
    --finetune-jobs finetune_jobs.json

# 6. Monitor in separate terminals
tail -f base_training_log.json
tail -f finetune_training_log.json

# 7. When complete, verify results
python3 -c "
import json
with open('finetune_training_log.json') as f:
    log = json.load(f)
    print(f'Total jobs: {log[\"total_jobs\"]}')
    print(f'Successful: {log[\"successful_jobs\"]}')
    print(f'Failed: {log[\"failed_jobs\"]}')
"
```

Good luck! 🚀
