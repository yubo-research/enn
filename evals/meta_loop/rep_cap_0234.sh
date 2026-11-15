#!/bin/bash
# Usage: rep_cap_0234.sh OUTDIR D MAXN CAP SECONDS REPS   (run from repo root)
# Replicated bytes_per_add.py runs under a cgroup cap, 1 CPU, store on the root disk.
O=$1; D=$2; MAXN=$3; C=$4; S=$5; R=$6; mkdir -p $O; O=$(realpath $O)
H=$(cd "$(dirname "$0")" && pwd)/bytes_per_add.py
for r in $(seq 1 $R); do
  W=/home/dsweet/bench_work/rep0234_$C; rm -rf $W; F=$O/bpa_D${D}_${C}_r$r.jsonl; rm -f $F
  PYTHONPATH=src timeout $S systemd-run --user --scope -q -p MemoryMax=$C -p MemoryHigh=$C -p CPUQuota=100% \
    python $H $D $MAXN $W $F nvme0n1p1 > ${F%.jsonl}.log 2>&1
  rm -rf $W
done
echo DONE >> $O/sweep.log
