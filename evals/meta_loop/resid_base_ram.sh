#!/bin/bash
# Usage: resid_base_ram.sh OUT D N REPS BASE HEAD   (repo root; 1G cap, 1 CPU)   report 0238
# Default-path (8-bit) in-RAM check: arms base (overlay dir BASE, e.g. wheel of b4218de), head (HEAD, refine-more on)
# and headn (HEAD, ENN_REFINE_MORE=0), order rotated per seed. BASE/HEAD are PYTHONPATH dirs holding enn/ and
# ennbo.libs/. Label in field "advise" (c4_base, c4_head, c4_headn). Stats: resid_ram_stats.py OUT (vs c4_base).
O=$1; D=$2; N=$3; R=$4; BASE=$5; HEAD=$6; H=$(cd "$(dirname "$0")" && pwd)/advise_data_ab.py
ARMS=(base head headn)
for s in $(seq 1 $R); do for j in 0 1 2; do
  a=${ARMS[$(( (s + j) % 3 ))]}; P=$HEAD; rm=1; [ $a = base ] && P=$BASE; [ $a = headn ] && rm=0
  W=/home/dsweet/bench_work/rram; rm -rf $W
  (cd /tmp && LD_LIBRARY_PATH=$P/ennbo.libs timeout 900 systemd-run --user --scope -q -p MemoryMax=1G -p MemoryHigh=1G \
    -p CPUQuota=100% -E ENN_DATA_ADVISE=c4_$a -E ENN_CODE4=0 -E ENN_REFINE_MORE=$rm -E SEED=$s -E PYTHONPATH=$P python $H $D $N $W $O)
  rm -rf $W
done; done
