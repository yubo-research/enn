#!/bin/bash
# Usage: pbuf_cold.sh OUTDIR D MAXN CAP SECONDS ARM...   (run from repo root; report 0242)
# True on-disk cost per add/query (cold_cost.py: store paged out before every op) per ENN_PBUF arm,
# sequential, cgroup cap, 1 CPU, store on the root disk. env INTRINSIC is not used by cold_cost.py (uniform sine data).
O=$1; D=$2; MAXN=$3; C=$4; S=$5; shift 5; mkdir -p $O; O=$(realpath $O)
H=$(cd "$(dirname "$0")" && pwd)/cold_cost.py
for a in "$@"; do
  W=/home/dsweet/bench_work/pbcold_$a; rm -rf $W; F=$O/cold_D${D}_${C}_pb$a.jsonl; rm -f $F
  timeout $S systemd-run --user --scope -q -p MemoryMax=$C -p MemoryHigh=$C -p CPUQuota=100% \
    -E ENN_PBUF=$a -E PYTHONPATH=${PP:-src} python $H $D $MAXN $W $F nvme0n1p1 > ${F%.jsonl}.log 2>&1
  rm -rf $W
  echo DONE cold $a $(date +%T) >> $O/sweep.log
done
