#!/bin/bash
# Usage: cold_cost.sh OUTDIR D MAXN CAP SECONDS   (run from repo root)
# True on-disk cost per add/query (cold_cost.py) under a cgroup cap, 1 CPU, store on the root disk.
O=$1; D=$2; MAXN=$3; C=$4; S=$5; mkdir -p $O; O=$(realpath $O)
H=$(cd "$(dirname "$0")" && pwd)/cold_cost.py
W=/home/dsweet/bench_work/cold_${D}_$C; rm -rf $W; F=$O/cold_D${D}_${C}.jsonl; rm -f $F
PYTHONPATH=src timeout $S systemd-run --user --scope -q -p MemoryMax=$C -p MemoryHigh=$C -p CPUQuota=100% \
  python $H $D $MAXN $W $F nvme0n1p1 > ${F%.jsonl}.log 2>&1
rm -rf $W
echo DONE $D $C >> $O/sweep.log
