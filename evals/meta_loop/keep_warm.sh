#!/bin/bash
# Usage: keep_warm.sh OUTDIR D N CAP SECONDS   (run from repo root)   report 0236
O=$1; D=$2; N=$3; C=$4; S=$5; mkdir -p $O; O=$(realpath $O)
H=$(cd "$(dirname "$0")" && pwd)/keep_warm.py
W=/home/dsweet/bench_work/keep_${D}_$N; rm -rf $W; F=$O/keep_D${D}_N${N}.jsonl; rm -f $F
PYTHONPATH=src timeout $S systemd-run --user --scope -q -p MemoryMax=$C -p MemoryHigh=$C -p CPUQuota=100% \
  python $H $D $N $W $F nvme0n1p1 > ${F%.jsonl}.log 2>&1
rm -rf $W
echo DONE $D $N $C >> $O/sweep.log
