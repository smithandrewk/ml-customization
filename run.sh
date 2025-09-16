#!/bin/bash
source env/bin/activate

start=${1:-0}
end=${2:-7}
device=${3:-0}
prefix=${4:-augmented_medium}
use_augmentation=${5:-true}

for i in $(seq $start $end)
do
  echo "Starting fold $i on device $device with prefix $prefix"
  if [ "$use_augmentation" = true ] ; then
    python3 train.py --fold $i --device $device -b 64 --model medium --use_augmentation --prefix $prefix
  else
    python3 train.py --fold $i --device $device -b 64 --model medium --prefix $prefix
  fi
done