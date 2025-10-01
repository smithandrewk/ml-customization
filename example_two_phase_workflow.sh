#!/bin/bash
# Complete example workflow for two-phase distributed training
# This demonstrates batch composition hyperparameter search

echo "=========================================="
echo "Two-Phase Training Example Workflow"
echo "=========================================="
echo ""

# Step 1: Generate experiment jobs
echo "Step 1: Generating experiment jobs..."
python3 << 'EOF'
import json

# Search over batch compositions
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
            'early_stopping_patience': 40,
            'early_stopping_patience_target': 40,
            'data_path': 'data/001_60s_window',
            'participants': participants,
            'prefix': f'batch_comp_pct{int(target_batch_pct*100)}',
            'n_base_participants': 'all',
            'target_data_pct': 1.0,
            'window_size': 3000,
        }
        jobs.append(job)

with open('jobs_config.json', 'w') as f:
    json.dump(jobs, f, indent=2)

print(f"Generated {len(jobs)} jobs")
print(f"  Batch compositions: {batch_compositions}")
print(f"  Folds: {len(participants)}")
EOF

echo ""
echo "✓ Created jobs_config.json with 15 experiment jobs"
echo ""

# Step 2: Analyze base model requirements
echo "Step 2: Analyzing base model requirements..."
python3 distributed_train_helper.py jobs_config.json

echo ""

# Step 3: Show what will happen
echo "=========================================="
echo "What will happen:"
echo "=========================================="
echo ""
echo "Phase 1: Train Base Models"
echo "  • All 15 experiments share the SAME base model config"
echo "  • Only 1 base model will be trained (distributed)"
echo "  • Base model copied to: experiments/base_models/{hash}/"
echo ""
echo "Phase 2: Run Experiments"
echo "  • 15 experiments will run (distributed)"
echo "  • Each loads the cached base model from Phase 1"
echo "  • Only target fine-tuning happens (fast!)"
echo "  • Results saved to: experiments/batch_comp_pct*/"
echo ""
echo "Compute savings: ~14 base training runs eliminated!"
echo ""

# Step 4: Ready to run
echo "=========================================="
echo "Ready to run!"
echo "=========================================="
echo ""
echo "To execute the two-phase training, run:"
echo ""
echo "  python run_two_phase_training.py \\"
echo "      --cluster-config cluster_config.json \\"
echo "      --jobs-config jobs_config.json \\"
echo "      --script-path oct1_train.py"
echo ""
echo "After completion, results will be in:"
echo "  • experiments/base_models/        (1 base model)"
echo "  • experiments/batch_comp_pct*/    (15 experiment results)"
echo "  • phase1_base_models.json         (Phase 1 execution log)"
echo "  • phase2_experiments.json         (Phase 2 execution log)"
echo ""
