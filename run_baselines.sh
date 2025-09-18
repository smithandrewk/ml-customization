#!/bin/bash

# Run baseline comparisons for smoking detection
# Usage: ./run_baselines.sh <fold> <device> <batch_size> [model_type]

if [ $# -lt 3 ]; then
    echo "Usage: $0 <fold> <device> <batch_size> [model_type]"
    echo "Example: $0 0 0 256 medium"
    exit 1
fi

FOLD=$1
DEVICE=$2
BATCH_SIZE=$3
MODEL_TYPE=${4:-medium}  # Default to medium if not specified

echo "Running baseline comparisons for fold $FOLD on device $DEVICE with batch size $BATCH_SIZE"
echo "Model type: $MODEL_TYPE"
echo "=========================================================================="

# 1. Generic Model Baseline
echo "1/4 Running Generic Model Baseline..."
python3 train_baseline.py \
    --fold $FOLD \
    --device $DEVICE \
    --batch_size $BATCH_SIZE \
    --model $MODEL_TYPE \
    --use_augmentation \
    --prefix "${MODEL_TYPE}_generic" \
    --mode generic

if [ $? -ne 0 ]; then
    echo "ERROR: Generic baseline failed!"
    exit 1
fi

echo "Generic baseline completed successfully!"
echo "=========================================================================="

# 2. Last-Layer Fine-tuning Baseline
echo "2/4 Running Last-Layer Fine-tuning Baseline..."
python3 train_baseline.py \
    --fold $FOLD \
    --device $DEVICE \
    --batch_size $BATCH_SIZE \
    --model $MODEL_TYPE \
    --use_augmentation \
    --prefix "${MODEL_TYPE}_last_layer" \
    --mode last_layer_only

if [ $? -ne 0 ]; then
    echo "ERROR: Last-layer baseline failed!"
    exit 1
fi

echo "Last-layer baseline completed successfully!"
echo "=========================================================================="

# 3. Target-Only Baseline
echo "3/4 Running Target-Only Baseline..."
python3 train_baseline.py \
    --fold $FOLD \
    --device $DEVICE \
    --batch_size $BATCH_SIZE \
    --model $MODEL_TYPE \
    --use_augmentation \
    --prefix "${MODEL_TYPE}_target_only" \
    --mode target_only

if [ $? -ne 0 ]; then
    echo "ERROR: Target-only baseline failed!"
    exit 1
fi

echo "Target-only baseline completed successfully!"
echo "=========================================================================="

# 4. Personalized Baseline (for comparison)
echo "4/4 Running Personalized Baseline (reference)..."
python3 train_baseline.py \
    --fold $FOLD \
    --device $DEVICE \
    --batch_size $BATCH_SIZE \
    --model $MODEL_TYPE \
    --use_augmentation \
    --prefix "${MODEL_TYPE}_personalized" \
    --mode personalized

if [ $? -ne 0 ]; then
    echo "ERROR: Personalized baseline failed!"
    exit 1
fi

echo "Personalized baseline completed successfully!"
echo "=========================================================================="

echo "All baselines completed for fold $FOLD!"
echo ""
echo "Experiment directories created:"
echo "- experiments/${MODEL_TYPE}_generic_*/"
echo "- experiments/${MODEL_TYPE}_last_layer_*/"
echo "- experiments/${MODEL_TYPE}_target_only_*/"
echo "- experiments/${MODEL_TYPE}_personalized_*/"
echo ""
echo "Next steps:"
echo "1. Run for all folds (0-7)"
echo "2. Use experiment_summary.py to compare results"
echo "3. Use compare_experiments.py for statistical analysis"