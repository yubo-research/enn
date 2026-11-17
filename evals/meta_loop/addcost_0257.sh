#!/bin/bash
O=$(realpath $1); shift
H=$(cd "$(dirname "$0")" && pwd)/addcost_0257.py
B=/home/dsweet/bench_work/a0257
for rep in $(seq ${REPS:-2}); do
  for spec in "$@"; do
    IFS=: read arm m full every lloyd pages graph <<< "$spec"
    W=$B/w_$arm; rm -rf $W; mkdir -p $W
    systemd-run --user --scope -q -p MemoryMax=1G -p MemoryHigh=1G -p CPUQuota=100% \
      env PYTHONPATH=${PP:-/home/dsweet/bench_work/ov_0257g} ARM=$arm ENN_REASSIGN=$m ENN_REASSIGN_FULL=$full \
      ENN_REASSIGN_EVERY=$every ENN_LLOYD_EVERY=$lloyd ENN_LLOYD_PAGES=$pages ENN_LLOYD_GRAPH=${graph:-0} \
      python $H ${D:-10} ${N:-1e5} $W $O
    rm -rf $W
  done
done
