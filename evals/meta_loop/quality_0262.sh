#!/bin/bash
# Usage: quality_0262.sh OUTDIR D N SEEDS "name:lloyd_every:lloyd_graph:walk_graph:leaf_coef:pbuf:lloyd_flush:tag ..."   (report 0262)
# As quality_0259.sh (paired-seed quality via recall_cmp.py, 2G scope, 1 CPU) with buffer fields as capq_0262.sh (tag 1 = ENN_PBUF_TAG=1).
# Overlay OV (default ov_0262b). Arm names must be integers (reassign_stats_0252.py). Stats: python reassign_stats_0252.py OUTDIR/..
O=$1; D=$2; N=$3; S=$4; ARMS=$5; mkdir -p $O; H=$(cd "$(dirname "$0")" && pwd)
W=${QW:-/mnt/scratch}/q0262_$$
for s in $(seq ${FIRST:-0} $((${FIRST:-0}+S-1))); do for a in $ARMS; do
  IFS=: read name lloyd graph walk leaf pbuf flush tag <<< "$a"
  on=$([[ $pbuf == 0 ]] && echo 0 || echo 1)
  rm -rf $W
  timeout 3000 systemd-run --user --scope -q -p MemoryMax=2G -p MemoryHigh=2G -p CPUQuota=100% \
    -E INTRINSIC=${INTRINSIC:-0} -E ENN_LLOYD_EVERY=$lloyd -E ENN_LLOYD_GRAPH=$graph -E ENN_WALK_GRAPH=$walk \
    -E ENN_PBUF=$pbuf -E ENN_LOG_PACE=1 -E ENN_PBUF_ALL=$on -E ENN_PBUF_PREFETCH=$on -E ENN_PBUF_PUNCH=$on -E ENN_LLOYD_FLUSH=$flush -E ENN_PBUF_TAG=${tag:-0} ${XV:+-E $XV} \
    -E PYTHONPATH=/home/dsweet/bench_work/${OV:-ov_0262b} python $H/recall_cmp.py $D $N $s $leaf $W $O/r$name.jsonl > /dev/null 2>> $O/err.log
  rm -rf $W
done; done
echo DONE >> $O/err.log
