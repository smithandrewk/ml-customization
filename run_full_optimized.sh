#!/bin/bash

# Optimized Full Model Training
# Conservative strategy: Longest patience for most complex model

batch_size=256
patience=120  # Triple patience for most complex model
model=full
lr=5e-5  # Very reduced learning rate for full model

echo "🚀 FULL MODEL ABLATION - Optimized Training Strategy"
echo "Model: $model | Batch: $batch_size | Patience: $patience | LR: 5e-5 (very conservative)"
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

echo "🎉 Full model ablation completed!"
echo ""
echo "✅ NOTE: Using very conservative LR (5e-5) optimized for large model architecture."
echo ""
echo "📊 ARCHITECTURE ABLATION ANALYSIS:"
echo "=========================================================================="
echo "All three architectures have been trained with optimized strategies:"
echo "• Simple:  patience=40,  LR=3e-4 (proven optimal)"
echo "• Medium:  patience=80,  LR=1e-4 (longer training, lower LR)"
echo "• Full:    patience=120, LR=5e-5 (longest training, lowest LR)"
echo ""
echo "📈 Analysis commands:"
echo "1. Overall summary:"
echo "   python3 experiment_summary.py"
echo ""
echo "2. Pairwise comparisons:"
echo "   python3 compare_experiments.py arch_ablation_simple_optimized arch_ablation_medium_optimized"
echo "   python3 compare_experiments.py arch_ablation_simple_optimized arch_ablation_full_optimized"
echo "   python3 compare_experiments.py arch_ablation_medium_optimized arch_ablation_full_optimized"
echo ""
echo "3. Training dynamics analysis:"
echo "   python3 compare_experiments.py arch_ablation_simple_optimized arch_ablation_medium_optimized --analyze-best"