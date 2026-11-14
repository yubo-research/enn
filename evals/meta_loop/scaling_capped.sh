#!/bin/bash
# Usage: scaling_capped.sh LABEL CHECKOUT OUTDIR D MAXN MAXSECONDS  -- add-time scaling under a 1 GB cgroup cap
L=$1; C=$2; O=$3; D=$4; MAXN=$5; MS=$6; mkdir -p $O
W=/home/dsweet/bench_work/s_${L}_$D; rm -rf $W
(cd $C && QMAX_N=1e5 PYTHONPATH=src systemd-run --user --scope -q -p MemoryMax=1G \
  python /home/dsweet/projects/enn/scripts/bench_cmp/bench.py $D $MAXN $W $O/${L}_D$D.jsonl $MS > $O/${L}_D$D.log 2>&1)
rm -rf $W
