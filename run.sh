batch_size=64
patience=40
for fold in {0..7}
do
    echo "Training fold $fold"
    python3 train.py \
        --fold $fold \
        --device 0 \
        --batch_size $batch_size \
        --model simple \
        --use_augmentation \
        --prefix simple_b${batch_size}_aug_patience${patience} \
        --early_stopping_patience $patience \
        --early_stopping_patience_target $patience \
        --mode last_layer_only
done