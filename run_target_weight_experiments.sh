TARGET_WEIGHTS=(0.0 1.0 2.0 5.0 10.0)
PARTICIPANTS=("ejaz" "asfik" "tonmoy")

for WEIGHT in "${TARGET_WEIGHTS[@]}"; do
    for PART in "${PARTICIPANTS[@]}"; do
        echo "Running experiment with target participant: $PART and target weight: $WEIGHT"
        python3 train.py --dataset_dir data/002_test --model simple --target_participant "$PART" --early_stopping_metric loss --target_weight "$WEIGHT"
    done
done