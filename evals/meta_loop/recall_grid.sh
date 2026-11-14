#!/bin/bash
# Usage: recall_grid.sh OUTDIR  -- recall@10 + quality vs leaf budget, under a 1 GB cgroup cap.
# Run from the repo root after `make build-ext`.
O=$1; mkdir -p $O; O=$(realpath $O)
R=$(cd "$(dirname "$0")" && pwd)/recall_cmp.py
run() {  # D N SEED LEAF
  W=/home/dsweet/bench_work/rc_$1_$2_$3_$4; rm -rf $W
  PYTHONPATH=src systemd-run --user --scope -q -p MemoryMax=1G python $R $1 $2 $3 $4 $W $O/recall.jsonl >>$O/recall.log 2>&1
  rm -rf $W
}
# Default leaf budget (10 * ln N): 3 sizes x 2 dims x 10 seeds.
for D in 10 100; do for N in 1e3 1e4 1e5; do for S in 0 1 2 3 4 5 6 7 8 9; do
  run $D $N $S 10
done; done; done
# Leaf budget sweep at N=1e5: 3, 30, 100 (5 seeds each).
for D in 10 100; do for L in 3 30 100; do for S in 0 1 2 3 4; do
  run $D 1e5 $S $L
done; done; done
echo DONE >>$O/recall.log
