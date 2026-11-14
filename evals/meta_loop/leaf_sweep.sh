#!/bin/bash
# Usage: leaf_sweep.sh OUTDIR -- recall@10 + paired quality vs query leaf budget (L * ln N) at N=1e5, seeds 0-9, 1 GB cap.
O=$1; mkdir -p $O; O=$(realpath $O)
R=$(cd "$(dirname "$0")" && pwd)/recall_cmp.py
for D in 10 100; do for L in 30 100; do for S in 0 1 2 3 4 5 6 7 8 9; do
  W=/home/dsweet/bench_work/ls_${D}_${L}_$S; rm -rf $W
  PYTHONPATH=src systemd-run --user --scope -q -p MemoryMax=1G python $R $D 1e5 $S $L $W $O/leaf.jsonl >>$O/leaf.log 2>&1
  rm -rf $W
done; done; done
echo DONE >>$O/leaf.log
