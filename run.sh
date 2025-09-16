#!/bin/bash
source env/bin/activate

start=${1:-0}
end=${2:-7}
device=${3:-0}

for i in $(seq $start $end)
do
  python3 train.py --fold $i --device $device -b 64 --model medium
done