#!/bin/bash
# Usage: single_add_0248.sh OUT D N0 M REPS   (repo root; report 0248). Geometric vs linear train_x growth, single-row adds,
# 1G scope, 1 CPU, EBS work dir, every-level buffers R=3000 (arm 3000aph), alternating arms per seed.
mkdir -p "$(dirname "$1")"; O=$(realpath -m $1); D=$2; N0=$3; M=$4; R=$5; H=$(cd "$(dirname "$0")" && pwd)/single_add_0248.py
for s in $(seq 1 $R); do for g in geo lin; do
  W=/home/dsweet/bench_work/sadd; rm -rf $W
  timeout 1800 systemd-run --user --scope -q -p MemoryMax=1G -p MemoryHigh=1G -p CPUQuota=100% \
    -E ARM=$g -E SEED=$s -E ENN_LOG_GROW_LINEAR=$([ $g = lin ] && echo 1 || echo 0) -E ENN_PBUF=3000 -E ENN_PBUF_ALL=1 \
    -E PYTHONPATH=$(realpath ${PP:-src}) python $H $D $N0 $M $W $O
  rm -rf $W
done; done
