#!/bin/bash
O=$1; D=$2; N=$3; S=$4; ARMS=$5; mkdir -p $O; H=$(cd "$(dirname "$0")" && pwd)
W=${QW:-/mnt/scratch}/q0257_$$
for s in $(seq ${FIRST:-0} $((${FIRST:-0}+S-1))); do for a in $ARMS; do
  IFS=: read name lloyd graph <<< "$a"
  rm -rf $W
  timeout 3000 systemd-run --user --scope -q -p MemoryMax=2G -p MemoryHigh=2G -p CPUQuota=100% \
    -E INTRINSIC=${INTRINSIC:-0} -E ENN_LLOYD_EVERY=$lloyd -E ENN_LLOYD_GRAPH=$graph \
    -E PYTHONPATH=${PP:-/home/dsweet/bench_work/ov_0257g} python $H/recall_cmp.py $D $N $s 0 $W $O/r$name.jsonl > /dev/null 2>> $O/err.log
  rm -rf $W
done; done
echo DONE >> $O/err.log
