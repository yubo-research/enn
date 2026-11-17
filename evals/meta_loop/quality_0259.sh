#!/bin/bash
# Usage: quality_0259.sh OUTDIR D N SEEDS "name:lloyd_every:lloyd_graph:walk_graph:leaf_coef:overlay[:VAR=VAL] ..."   (report 0259)
# As quality_0258.sh, but each arm names its own overlay (dir under /home/dsweet/bench_work; empty = $PP or ov_0259a),
# so old and new builds alternate within each seed; an optional 7th field passes one extra env var (e.g. ENN_GRAPH_PRUNE=1). leaf_coef 0 = default 100 lnN. Paired-seed quality + in-RAM timing
# (recall_cmp.py), 2G scope, 1 CPU. Stats: python reassign_stats_0252.py OUTDIR/..
O=$1; D=$2; N=$3; S=$4; ARMS=$5; mkdir -p $O; H=$(cd "$(dirname "$0")" && pwd)
W=${QW:-/mnt/scratch}/q0259_$$
for s in $(seq ${FIRST:-0} $((${FIRST:-0}+S-1))); do for a in $ARMS; do
  IFS=: read name lloyd graph walk leaf ov xv <<< "$a"
  pp=/home/dsweet/bench_work/${ov:-${PP_NAME:-ov_0259a}}
  rm -rf $W
  timeout 3000 systemd-run --user --scope -q -p MemoryMax=2G -p MemoryHigh=2G -p CPUQuota=100% \
    -E INTRINSIC=${INTRINSIC:-0} -E ENN_LLOYD_EVERY=$lloyd -E ENN_LLOYD_GRAPH=$graph -E ENN_WALK_GRAPH=${walk:-0} ${xv:+-E $xv} \
    -E PYTHONPATH=$pp python $H/recall_cmp.py $D $N $s ${leaf:-0} $W $O/r$name.jsonl > /dev/null 2>> $O/err.log
  rm -rf $W
done; done
echo DONE >> $O/err.log
