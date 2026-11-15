#!/bin/bash
# Usage: long_0224.sh OUTDIR CAP MAXN TIMEOUT_S -- sorted-run (mrg) buffers, F=16, one at a time.
O=$1; C=$2; MAXN=$3; T=$4; H=$(cd "$(dirname "$0")" && pwd)
for c in 1024 4096; do
  d=$O/long_c${c}_f16; mkdir -p $d
  bash $H/buffer_sim.sh $d $C $MAXN $T buffered $c 16 mrg
done
echo ok > $O/long_done.txt
