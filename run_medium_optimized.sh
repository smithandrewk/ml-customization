#!/bin/bash

# Optimized Medium Model Training
# Adjusted strategy: Lower LR + Longer patience for larger model

batch_size=256
patience=80  # Doubled patience for more complex model
model=medium
lr=1e-4  # Reduced learning rate for medium model

echo "🚀 MEDIUM MODEL ABLATION - Optimized Training Strategy"
echo "Model: $model | Batch: $batch_size | Patience: $patience | LR: 1e-4 (reduced)"
echo "=========================================================================="

for fold in {0..7}
do
    participant_names=('alsaad' 'anam' 'asfik' 'ejaz' 'iftakhar' 'tonmoy' 'unk1' 'dennis')
    participant=${participant_names[$fold]}

    echo "📊 Training fold $fold (Participant: $participant)"
    echo "------------------------------------------------------------------------"

    python3 train.py \
        --fold $fold \
        --device 0 \
        --batch_size $batch_size \
        --model $model \
        --use_augmentation \
        --prefix "arch_ablation_${model}_optimized" \
        --early_stopping_patience $patience \
        --early_stopping_patience_target $patience \
        --mode full_fine_tuning \
        --lr $lr

    if [ $? -eq 0 ]; then
        echo "✅ Fold $fold completed successfully"
    else
        echo "❌ Fold $fold failed!"
    fi
    echo ""
done

echo "🎉 Medium model ablation completed!"
echo ""
echo "✅ NOTE: Using optimized LR (1e-4) for medium model architecture."
echo ""
echo "📈 Next steps:"
echo "1. Run full model: ./run_full_optimized.sh"
echo "2. Compare all architectures: python3 experiment_summary.py"
echo "3. Statistical comparison:"
echo "   python3 compare_experiments.py arch_ablation_simple_optimized arch_ablation_medium_optimized"