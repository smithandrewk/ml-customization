#!/bin/bash

# Run all baseline experiments across all folds
# Usage: ./run_all_baselines.sh <device> <batch_size> [model_type]

if [ $# -lt 2 ]; then
    echo "Usage: $0 <device> <batch_size> [model_type]"
    echo "Example: $0 0 256 medium"
    exit 1
fi

DEVICE=$1
BATCH_SIZE=$2
MODEL_TYPE=${3:-medium}  # Default to medium if not specified

echo "Running complete baseline experiment suite"
echo "Device: $DEVICE, Batch size: $BATCH_SIZE, Model: $MODEL_TYPE"
echo "This will run 4 baselines × 8 folds = 32 training runs"
echo "=========================================================================="

# Array of all folds (participant indices)
FOLDS=(0 1 2 3 4 5 6 7)

# Track failed runs
FAILED_RUNS=()

for fold in "${FOLDS[@]}"; do
    echo ""
    echo "🔄 STARTING FOLD $fold (Participant: $(python3 -c "participants=['alsaad','anam','asfik','ejaz','iftakhar','tonmoy','unk1','dennis']; print(participants[$fold])"))"
    echo "========================================================================"

    # Run all baselines for this fold
    ./run_baselines.sh $fold $DEVICE $BATCH_SIZE $MODEL_TYPE

    if [ $? -ne 0 ]; then
        echo "❌ FOLD $fold FAILED!"
        FAILED_RUNS+=("fold_$fold")
    else
        echo "✅ FOLD $fold COMPLETED SUCCESSFULLY!"
    fi

    echo "========================================================================"
done

echo ""
echo "🎉 EXPERIMENT SUITE COMPLETED!"
echo ""

if [ ${#FAILED_RUNS[@]} -eq 0 ]; then
    echo "✅ All runs completed successfully!"
else
    echo "❌ Some runs failed:"
    for failed in "${FAILED_RUNS[@]}"; do
        echo "  - $failed"
    done
    echo ""
    echo "You may need to re-run the failed folds manually."
fi

echo ""
echo "📊 ANALYSIS COMMANDS:"
echo ""
echo "1. Quick summary of all experiments:"
echo "   python3 experiment_summary.py"
echo ""
echo "2. Compare specific baselines:"
echo "   python3 compare_experiments.py ${MODEL_TYPE}_personalized ${MODEL_TYPE}_generic"
echo "   python3 compare_experiments.py ${MODEL_TYPE}_personalized ${MODEL_TYPE}_last_layer"
echo "   python3 compare_experiments.py ${MODEL_TYPE}_personalized ${MODEL_TYPE}_target_only"
echo ""
echo "3. Statistical analysis with training dynamics:"
echo "   python3 compare_experiments.py ${MODEL_TYPE}_personalized ${MODEL_TYPE}_generic --analyze-best"
echo ""
echo "Results are saved in experiments/ directory with prefixes:"
echo "- ${MODEL_TYPE}_generic_*"
echo "- ${MODEL_TYPE}_last_layer_*"
echo "- ${MODEL_TYPE}_target_only_*"
echo "- ${MODEL_TYPE}_personalized_*"