#!/bin/bash
# Usage: screen_0264.sh OUT D N SEED   (report 0264)
# Builds the unbuffered graph arm (g10: LLOYD_GRAPH, EVERY=10, WALK_GRAPH=2) in RAM, dumps the tree,
# XV="VAR=val ..." adds store flags (e.g. the buffered bg10t arm). Then simulates re-parenting (internal-level Lloyd) offline: tree-descent owner rank before/after.
O=$1; D=$2; N=$3; SEED=$4; H=$(cd "$(dirname "$0")" && pwd)
W=/home/dsweet/bench_work/screen0264_${D}_${N}_${SEED}; rm -rf $W
systemd-run --user --scope -q -p MemoryMax=3G -p MemoryHigh=3G -p CPUQuota=100% \
  -E ENN_LLOYD_EVERY=10 -E ENN_LLOYD_GRAPH=1 -E ENN_WALK_GRAPH=2 ${ARMS:+-E "ARMS=$ARMS"} $(for v in $XV; do echo -n "-E $v "; done) -E NP=${NP:-2000} -E WIDE=${WIDE:-16} \
  -E PYTHONPATH=/home/dsweet/bench_work/${OV:-ov_0263f} python $H/screen_0264.py $D $N $SEED $W $O
rm -rf $W
