#!/bin/bash
# Usage: long_0223.sh OUTDIR MAXN TIMEOUT_S -- run-indexed buffers (idx), 256M cap, F=16, one at a time.
O=$1; MAXN=$2; T=$3; H=$(cd "$(dirname "$0")" && pwd)
for c in 4096 16384; do
  d=$O/long_c${c}_f16; mkdir -p $d
  bash $H/buffer_sim.sh $d 256M $MAXN $T buffered $c 16 idx
done
echo ok > $O/long_done.txt
