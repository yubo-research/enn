#!/bin/bash
# Usage: gwalk_0258.sh OUTDIR N:SEED:MODE ...   (report 0258) leaves-to-last-true-neighbour for the graph-ordered walk
# PATIENCE=P sets the patience stop (P lnN quiet leaves; 0 = off). On the leaf-graph Lloyd partition (ENN_LLOYD_GRAPH=1, ENN_LLOYD_EVERY=${EVERY:-10}); MODE = ENN_WALK_GRAPH (0 off, 1 graph-first, 2 merged).
O=$1; shift; mkdir -p $O; O=$(realpath $O)
H=$(cd "$(dirname "$0")" && pwd)/growth_0256.py
B=/home/dsweet/bench_work/g0258; mkdir -p $B
PP=${PP:-/home/dsweet/bench_work/ov_0258}
for spec in "$@"; do
  IFS=: read n seed mode <<< "$spec"
  arm=gw${mode}e${EVERY:-10}p${PATIENCE:-0}
  W=$B/w_${arm}_${n}_${seed}; rm -rf $W
  PYTHONPATH=$PP ARM=$arm CAP=64 OFFLINE= LLOYD=0 NQ=${NQ:-300} ENN_LLOYD_GRAPH=1 ENN_LLOYD_EVERY=${EVERY:-10} \
    ENN_WALK_GRAPH=$mode PATIENCE=${PATIENCE:-0} python $H ${D:-10} $n $seed $W $O/gw_${arm}.jsonl > $O/gw_${arm}_${n}_${seed}.log 2>&1
  rm -rf $W
  echo "DONE $arm $n $seed $(date +%T)" >> $O/sweep.log
done
echo ALLDONE $(date +%T) >> $O/sweep.log
