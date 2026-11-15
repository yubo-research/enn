#!/bin/bash
# Usage: parent_count.sh OUTDIR MAXN   (repo root; report 0242)
# Per-query leaf / leaf-parent / internal-node counts (parent_count.py) for D=10, D=100, D=100 INTRINSIC=5; 1G cap, 1 CPU.
O=$1; MAXN=$2; mkdir -p $O; O=$(realpath $O); H=$(cd "$(dirname "$0")" && pwd)/parent_count.py
for arm in "10 0" "100 0" "100 5"; do
  set -- $arm; W=/home/dsweet/bench_work/pc_$1_$2; rm -rf $W; F=$O/pc_D$1_i$2.jsonl; rm -f $F
  PYTHONPATH=src timeout 3600 systemd-run --user --scope -q -p MemoryMax=1G -p MemoryHigh=1G -p CPUQuota=100% \
    -E INTRINSIC=$2 python $H $1 $MAXN $W $F > ${F%.jsonl}.log 2>&1
  rm -rf $W
done
echo DONE >> $O/sweep.log
