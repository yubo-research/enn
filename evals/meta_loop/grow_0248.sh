#!/bin/bash
# Usage: grow_0248.sh OUTDIR CAP MAXN REPS ARM   (repo root; report 0248)
# train_x/train_y file growth A/B: geometric (default) vs linear (ENN_LOG_GROW_LINEAR=1), alternated per rep,
# stall_probe.sh D=100 WINDOW=1e6 -> OUTDIR/rep<i>/{geo,lin}/stall_D100_<CAP>_pb<ARM>.jsonl
O=$1; C=$2; M=$3; R=$4; A=$5
for r in $(seq 1 $R); do for g in geo lin; do
  LOG_LINEAR=$([ $g = lin ] && echo 1 || echo 0) WINDOW=1e6 SECS=${SECS:-5400} bash "$(dirname "$0")"/stall_probe.sh $O/rep$r/$g 100 $M $C $A
done; done
