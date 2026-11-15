#!/bin/bash
# Usage: leaf_ids_ab.sh OUTDIR D MAXN CAP SECONDS   (run from repo root)   report 0236
O=$1; D=$2; N=$3; C=$4; S=$5; mkdir -p $O; O=$(realpath $O)
H=$(cd "$(dirname "$0")" && pwd)/leaf_ids_ab.py
W=/home/dsweet/bench_work/lids_${D}; rm -rf $W; F=$O/lids_D${D}_${C}.jsonl; rm -f $F
PYTHONPATH=src timeout $S systemd-run --user --scope -q -p MemoryMax=$C -p MemoryHigh=$C -p CPUQuota=100% \
  python $H $D $N $W $F nvme0n1p1 > ${F%.jsonl}.log 2>&1
rm -rf $W
echo DONE $D $N $C >> $O/sweep.log
