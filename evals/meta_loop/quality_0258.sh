#!/bin/bash
# Usage: quality_0258.sh OUTDIR D N SEEDS "name:lloyd_every:lloyd_graph:walk_graph[:leaf_coef] ..."  (leaf_coef 0 = default 100 lnN)   (report 0258)
# Paired-seed quality + in-RAM timing (recall_cmp.py), 2G scope, 1 CPU; arms alternate within each seed.
O=$1; D=$2; N=$3; S=$4; ARMS=$5; mkdir -p $O; H=$(cd "$(dirname "$0")" && pwd)
W=${QW:-/mnt/scratch}/q0258_$$
for s in $(seq ${FIRST:-0} $((${FIRST:-0}+S-1))); do for a in $ARMS; do
  IFS=: read name lloyd graph walk leaf <<< "$a"
  rm -rf $W
  timeout 3000 systemd-run --user --scope -q -p MemoryMax=2G -p MemoryHigh=2G -p CPUQuota=100% \
    -E INTRINSIC=${INTRINSIC:-0} -E ENN_LLOYD_EVERY=$lloyd -E ENN_LLOYD_GRAPH=$graph -E ENN_WALK_GRAPH=${walk:-0} \
    -E PYTHONPATH=${PP:-/home/dsweet/bench_work/ov_0258} python $H/recall_cmp.py $D $N $s ${leaf:-0} $W $O/r$name.jsonl > /dev/null 2>> $O/err.log
  rm -rf $W
done; done
echo DONE >> $O/err.log
