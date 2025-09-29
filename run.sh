batch_size=512
patience=5
lr=3e-4
mode="target_only" # 'full_fine_tuning', 'last_layer_only', 'generic', 'target_only'
timestamp=$(date +%Y%m%d_%H%M%S)
prefix="b${batch_size}_aug_patience${patience}_${mode}_${timestamp}"

for fold in {0..2}
do
    echo "Training fold $fold"
    python3 train.py \
        --fold $fold \
        --device 0 \
        --batch_size $batch_size \
        --model test \
        --use_augmentation \
        --prefix $prefix \
        --early_stopping_patience $patience \
        --early_stopping_patience_target $patience \
        --mode $mode \
        --lr $lr
done

scp -r experiments/${prefix} 10.173.98.188:~/ml-customization/experiments/
rm -rf experiments/${prefix}