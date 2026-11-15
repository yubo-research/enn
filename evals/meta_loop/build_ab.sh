#!/bin/bash
# Usage: build_ab.sh OUT D N REPS BASE_PYTHONPATH   (repo root; 1G cap, 1 CPU)   report 0236
# Paired in-RAM add/query timing (advise_data_ab.py) of two builds: "base" (BASE_PYTHONPATH) vs "head" (src), alternating per seed.
O=$1; D=$2; N=$3; R=$4; B=$5; H=$(cd "$(dirname "$0")" && pwd)/advise_data_ab.py
for s in $(seq 1 $R); do for a in base head; do
  P=src; [ $a = base ] && P=$B
  W=/home/dsweet/bench_work/bab; rm -rf $W
  timeout 900 systemd-run --user --scope -q -p MemoryMax=1G -p MemoryHigh=1G -p CPUQuota=100% \
    -E ENN_DATA_ADVISE=$a -E SEED=$s -E PYTHONPATH=$P python $H $D $N $W $O
  rm -rf $W
done; done
