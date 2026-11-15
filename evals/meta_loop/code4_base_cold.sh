#!/bin/bash
# Usage: code4_base_cold.sh OUTDIR D MAXN CAP BASE_PP HEAD_PP   (repo root)   report 0237
# Cold A/B of two builds with ENN_CODE4=0 (default path): arm a0 = BASE_PP, a1 = HEAD_PP, one rep,
# same data/queries (code4_ab.py). Summarize with code4_stats.py OUTDIR D CAP 1.
O=$1; D=$2; N=$3; C=$4; mkdir -p $O; O=$(realpath $O); H=$(cd "$(dirname "$0")" && pwd)/code4_ab.py
for a in 0 1; do
  P=$5; [ $a = 1 ] && P=$6
  W=/home/dsweet/bench_work/c4b_${D}; rm -rf $W; F=$O/c4_D${D}_${C}_a${a}_r1.jsonl; rm -f $F
  (cd /tmp && PYTHONPATH=$P timeout 3600 systemd-run --user --scope -q -p MemoryMax=$C -p MemoryHigh=$C -p CPUQuota=100% \
    -E ENN_CODE4=0 python $H build_$a $D $N $W $F nvme0n1p1 > ${F%.jsonl}.log 2>&1)
  rm -rf $W
done
