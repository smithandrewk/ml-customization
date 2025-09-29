batch_size=256
patience=40
lr=3e-4

for fold in {0..0}
do
    echo "Training fold $fold"
    python3 train.py \
        --fold $fold \
        --device 0 \
        --batch_size $batch_size \
        --model test \
        --use_augmentation \
        --prefix sep29/b${batch_size}_aug_patience${patience} \
        --early_stopping_patience $patience \
        --early_stopping_patience_target $patience \
        --mode full_fine_tuning \
        --lr $lr
done