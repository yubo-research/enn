#!/bin/bash
# Usage: main_quality_grid.sh CHECKOUT OUTDIR -- quality_cmp.py (loglik, nrmse) on another checkout
# (e.g. a git worktree of main with a built extension), same grid as recall_grid.sh default leaf.
C=$1; O=$2; mkdir -p $O; O=$(realpath $O); C=$(realpath $C)
R=$(cd "$(dirname "$0")" && pwd)/quality_cmp.py
for D in 10 100; do for N in 1e3 1e4 1e5; do for S in 0 1 2 3 4 5 6 7 8 9; do
  W=/home/dsweet/bench_work/mq_${D}_${N}_${S}; rm -rf $W
  (cd $C && PYTHONPATH=src systemd-run --user --scope -q -p MemoryMax=1G python $R $D $N $S $W $O/quality.jsonl >>$O/quality.log 2>&1)
  rm -rf $W
done; done; done
echo DONE >>$O/quality.log
