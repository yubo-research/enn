#!/bin/bash
# Usage: code4_ab.sh OUTDIR D MAXN CAP SECONDS REPS   (run from repo root)   report 0237
# 8-bit (ENN_CODE4=0) vs 4-bit (ENN_CODE4=1) page codes, one process per arm, order ABBA per pair of reps,
# PP=PYTHONPATH of the build under test (default src). Under a cgroup cap with 1 CPU and the store on the root disk (EBS). Rows: code4_ab.py (env INTRINSIC passes through via -E).
O=$1; D=$2; N=$3; C=$4; S=$5; R=${6:-2}; mkdir -p $O; O=$(realpath $O)
H=$(cd "$(dirname "$0")" && pwd)/code4_ab.py
for r in $(seq 1 $R); do
  ORDER="0 1"; [ $((r % 2)) = 0 ] && ORDER="1 0"
  for a in $ORDER; do
    W=/home/dsweet/bench_work/c4_${D}; rm -rf $W; F=$O/c4_D${D}_${C}_a${a}_r${r}.jsonl; rm -f $F
    PYTHONPATH=${PP:-src} timeout $S systemd-run --user --scope -q -p MemoryMax=$C -p MemoryHigh=$C -p CPUQuota=100% \
      -E ENN_CODE4=$a -E INTRINSIC=${INTRINSIC:-0} python $H c4_$a $D $N $W $F nvme0n1p1 > ${F%.jsonl}.log 2>&1
    rm -rf $W
  done
done
echo DONE $D $N $C >> $O/sweep.log
