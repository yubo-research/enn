#!/bin/bash
# Usage: quality_0266.sh OUTDIR D N SEEDS NAME   (report 0266; paired-seed quality of the current build via recall_cmp.py, 2G scope, 1 CPU)
# NAME must be an integer (reassign_stats_0252.py). Env: OV = wheel overlay under /home/dsweet/bench_work (default: repo src),
# LEAF (leaf coefficient; 0 keeps the library default), INTRINSIC, FIRST (first seed), QW (scratch dir). Stats: python reassign_stats_0252.py OUTDIR/..
O=$1; D=$2; N=$3; S=$4; NAME=$5; mkdir -p $O; H=$(cd "$(dirname "$0")" && pwd)
PP=${OV:+/home/dsweet/bench_work/$OV}; PP=${PP:-$(cd "$H/../.." && pwd)/src}
W=${QW:-/mnt/scratch}/q0266_$$
for s in $(seq ${FIRST:-0} $((${FIRST:-0}+S-1))); do
  rm -rf $W
  timeout 3000 systemd-run --user --scope -q -p MemoryMax=2G -p MemoryHigh=2G -p CPUQuota=100% \
    -E INTRINSIC=${INTRINSIC:-0} -E PYTHONPATH=$PP python $H/recall_cmp.py $D $N $s ${LEAF:-0} $W $O/r$NAME.jsonl > /dev/null 2>> $O/err.log
  rm -rf $W
done
echo DONE >> $O/err.log
