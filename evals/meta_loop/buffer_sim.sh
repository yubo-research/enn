#!/bin/bash
# Usage: buffer_sim.sh OUTDIR CAP MAXN TIMEOUT_S MODE [CAP_ROWS FANOUT [sub]]
# 'sub' bounds each node's buffer by its subtree capacity (see buffer_sim.py).
O=$1; C=$2; MAXN=$3; T=$4; M=$5; CR=${6:-1024}; F=${7:-16}; B=${8:-}
mkdir -p $O; O=$(realpath $O); H=$(cd "$(dirname "$0")" && pwd)/buffer_sim.py
W=/home/dsweet/bench_work/bs_${M}_$C; rm -rf $W
timeout $T systemd-run --user --scope -q -p MemoryMax=$C -p MemoryHigh=$C -p CPUQuota=100% \
  python $H $M $W $O/bs_${M}_${C}_c${CR}_f$F.jsonl $MAXN $CR $F nvme0n1p1 $B > $O/bs_${M}_${C}.log 2>&1
rm -rf $W
