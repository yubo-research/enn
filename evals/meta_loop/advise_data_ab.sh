#!/bin/bash
# Usage: advise_data_ab.sh OUT D N REPS   (repo root; 1G cap, 1 CPU; alternates normal/random)
O=$1; D=$2; N=$3; R=$4; H=$(cd "$(dirname "$0")" && pwd)/advise_data_ab.py
for s in $(seq 1 $R); do for a in normal random; do
  W=/home/dsweet/bench_work/adab; rm -rf $W
  timeout 900 systemd-run --user --scope -q -p MemoryMax=1G -p MemoryHigh=1G -p CPUQuota=100% \
    -E ENN_DATA_ADVISE=$a -E SEED=$s -E PYTHONPATH=src python $H $D $N $W $O
  rm -rf $W
done; done
