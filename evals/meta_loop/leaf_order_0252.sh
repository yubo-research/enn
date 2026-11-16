#!/bin/bash
# Usage: leaf_order_0252.sh OUT   (report 0252) offline leaf shape / visit-order screen, in RAM
# env LEAF_SPLIT=2 -> ENN_LEAF_SPLIT (two-means leaf split); TAG = work-dir suffix for concurrent runs; PP = overlay
# env NQ (default 200); SPECS="D:N:INTRINSIC:SEED ..."; PARTS, GRAPH_M pass through
O=$1; mkdir -p $O; H=$(cd "$(dirname "$0")" && pwd); W=/home/dsweet/bench_work/lo_w${TAG}
for spec in ${SPECS:-10:1e4:0:0 10:3e4:0:0 10:1e5:0:0 10:3e5:0:0 10:1e6:0:0 100:1e4:5:0 100:1e5:5:0 100:3e5:5:0}; do
  IFS=: read -r d n i s <<< "$spec"; rm -rf $W
  ENN_LEAF_SPLIT=${LEAF_SPLIT:-} INTRINSIC=$i PYTHONPATH=${PP:-/home/dsweet/projects/enn/src} timeout 5000 python $H/leaf_order_0252.py $d $n ${s:-0} $W $O/order.jsonl >> $O/summary.txt 2>&1
  rm -rf $W
done
echo DONE >> $O/summary.txt
