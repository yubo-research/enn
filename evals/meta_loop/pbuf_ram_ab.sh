#!/bin/bash
# Usage: pbuf_ram_ab.sh OUT D N REPS   (repo root; 1G cap, 1 CPU; report 0242)
# Paired in-RAM add/query timing (advise_data_ab.py, uniform data) of ENN_PBUF arms in $ARMS (default "0 1000")
# on one build (PP = PYTHONPATH, default src), order rotated per seed; arm label "pb_<arm>" in field "advise". Stats: pbuf_ram_stats.py OUT pb_0
O=$(realpath -m $1); D=$2; N=$3; R=$4; P=$(realpath ${PP:-src}); H=$(cd "$(dirname "$0")" && pwd)/advise_data_ab.py
read -ra A <<< "${ARMS:-0 1000}"
for s in $(seq 1 $R); do for j in $(seq 0 $((${#A[@]} - 1))); do
  a=${A[$(( (s + j) % ${#A[@]} ))]}
  W=/home/dsweet/bench_work/pbram; rm -rf $W
  (cd /tmp && timeout 900 systemd-run --user --scope -q -p MemoryMax=1G -p MemoryHigh=1G -p CPUQuota=100% \
    -E ENN_DATA_ADVISE=pb_$a -E ENN_PBUF=$a -E SEED=$s -E PYTHONPATH=$P python $H $D $N $W $O)
  rm -rf $W
done; done
