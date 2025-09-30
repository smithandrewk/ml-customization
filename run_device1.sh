batch_size=512
patience=40
lr=3e-4
mode="full_fine_tuning" # 'full_fine_tuning', 'last_layer_only', 'generic', 'target_only'
target_data_pcts=(0.05 0.5 1.0)  # Percentage of target training data to use (0.25, 0.5, 0.75, 1.0)
timestamp=$(date +%Y%m%d_%H%M%S)

for target_data_pct in "${target_data_pcts[@]}"
do
    prefix="b${batch_size}_aug_patience${patience}_${mode}_pct${target_data_pct}_${timestamp}"
    for fold in {0..7}
    do
        echo "Training fold $fold"
        python3 train.py \
            --fold $fold \
            --device 1 \
            --batch_size $batch_size \
            --model test \
            --use_augmentation \
            --prefix $prefix \
            --early_stopping_patience $patience \
            --early_stopping_patience_target $patience \
            --mode $mode \
            --lr $lr \
            --target_data_pct $target_data_pct
    done
    scp -r experiments/${prefix} 10.173.98.188:~/ml-customization/experiments/
    rm -rf experiments/${prefix}
done
