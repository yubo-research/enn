#!/bin/bash
# Usage: sweep_0221.sh OUTDIR MAXN TIMEOUT_S  -- runs buffer_sim.sh (buffered, 256M cap) over (CAP_ROWS,FANOUT) pairs, one at a time.
O=$1; MAXN=$2; T=$3; H=$(cd "$(dirname "$0")" && pwd)
for cf in "1024 16" "1024 64" "4096 16" "4096 64" "256 16" "1024 8"; do
  set -- $cf; d=$O/c$1_f$2; mkdir -p $d
  bash $H/buffer_sim.sh $d 256M $MAXN $T buffered $1 $2
done
