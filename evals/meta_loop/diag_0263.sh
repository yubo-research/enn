#!/bin/bash
# Usage: diag_0263.sh OUT D N SEED "name:pbuf:tag:rehome:leaf_rows ..."   (report 0263)
# Graph arm (LLOYD_GRAPH, EVERY=10, WALK_GRAPH=2, 10 lnN, LLOYD_FLUSH) + every-level buffers of R=pbuf rows.
# Per missed/found true neighbour: leaf / upper buffer / leaf-parent buffer, depth, owner slot scanned, owner leaf popped.
O=$1; D=$2; N=$3; SEED=$4; ARMS=$5; H=$(cd "$(dirname "$0")" && pwd)
for a in $ARMS; do
  IFS=: read name pbuf tag rehome lrows <<< "$a"
  on=$([[ $pbuf == 0 ]] && echo 0 || echo 1)
  W=/home/dsweet/bench_work/diag0263_$name; rm -rf $W
  systemd-run --user --scope -q -p MemoryMax=2G -p MemoryHigh=2G -p CPUQuota=100% \
    -E ENN_PBUF=$pbuf -E ENN_LOG_PACE=1 -E ENN_PBUF_ALL=$on -E ENN_PBUF_PREFETCH=$on -E ENN_PBUF_PUNCH=$on \
    -E ENN_LLOYD_EVERY=10 -E ENN_LLOYD_GRAPH=1 -E ENN_WALK_GRAPH=2 -E LEAF_COEF=${COEF:-10} -E ENN_LLOYD_FLUSH=$on -E ENN_PBUF_TAG=$tag -E ENN_PBUF_REHOME=${rehome:-0} -E ENN_PBUF_LEAF_ROWS=${lrows:-0} ${XV:+-E $XV} \
    -E ARM=$name -E NQ=${NQ:-300} -E PYTHONPATH=/home/dsweet/bench_work/${OV:-ov_0263f} python $H/diag_0263.py $D $N $SEED $W $O
  rm -rf $W
done
