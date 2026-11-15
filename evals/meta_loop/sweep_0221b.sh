#!/bin/bash
# Usage: sweep_0221b.sh OUTDIR MAXN TIMEOUT_S -- bounded-buffer ('sub') sweep, 256M cap, F=16.
O=$1; MAXN=$2; T=$3; H=$(cd "$(dirname "$0")" && pwd)
for c in 1024 4096 16384; do
  d=$O/sub_c${c}_f16; mkdir -p $d
  bash $H/buffer_sim.sh $d 256M $MAXN $T buffered $c 16 sub
done
