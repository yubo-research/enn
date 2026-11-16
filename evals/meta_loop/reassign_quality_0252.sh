#!/bin/bash
# Usage: reassign_quality_0252.sh OUT D N SEEDS "M1 M2 ..."   (report 0252)
# Paired recall/NRMSE/loglik (recall_cmp.py) per split-time reassignment size M (ENN_REASSIGN; 0 = off), 2G scope, 1 CPU.
# env: PP overlay (needs tree_reassign), INTRINSIC, FIRST (first seed, default 0), QW work-dir base, EVERY (ENN_REASSIGN_EVERY), FULL (ENN_REASSIGN_FULL).
# Writes OUT/r<M>.jsonl; compare with pair_0250.py OUT/r16.jsonl OUT/r0.jsonl (or main: report-0246-data/score/main.jsonl at D=10; report-0251-data/qual/d100i5_1e5/pmain.jsonl).
O=$1; D=$2; N=$3; S=$4; ARMS=$5; mkdir -p $O; H=$(cd "$(dirname "$0")" && pwd)
W=${QW:-/mnt/scratch}/rq_$$
for s in $(seq ${FIRST:-0} $((${FIRST:-0}+S-1))); do for m in $ARMS; do
  rm -rf $W
  timeout 3000 systemd-run --user --scope -q -p MemoryMax=2G -p MemoryHigh=2G -p CPUQuota=100% \
    -E INTRINSIC=${INTRINSIC:-0} -E ENN_REASSIGN=$m -E ENN_REASSIGN_EVERY=${EVERY:-0} -E ENN_REASSIGN_FULL=${FULL:-0} \
    -E PYTHONPATH=${PP:-/home/dsweet/projects/enn/src} python $H/recall_cmp.py $D $N $s 0 $W $O/r$m.jsonl > /dev/null 2>> $O/err.log
  rm -rf $W
done; done
echo DONE >> $O/err.log
