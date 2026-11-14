#!/bin/bash
# Usage: quality_grid.sh LABEL CHECKOUT OUTDIR   (runs under a 1 GB cgroup cap if systemd-run works)
L=$1; C=$2; O=$3; mkdir -p $O
for D in 10 100; do for N in 1e3 1e4 1e5; do for S in 0 1 2; do
  W=/home/dsweet/bench_work/q_${L}_$D; rm -rf $W
  (cd $C && PYTHONPATH=src systemd-run --user --scope -q -p MemoryMax=1G python /home/dsweet/projects/enn/evals/meta_loop/quality_cmp.py $D $N $S $W $O/$L.jsonl >>$O/$L.log 2>&1)
  rm -rf $W
done; done; done
