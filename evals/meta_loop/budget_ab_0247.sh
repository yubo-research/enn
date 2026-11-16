#!/bin/bash
# Usage: budget_ab_0247.sh OUTDIR D N SEEDS...   (repo root; report 0247)
# Paired timing+quality under the 1 GB / 1 CPU scope (quality_cmp.py) for three arms:
#   c30  = LEAF_COEF=30 INTERNAL_COEF=10 (pre-0247 defaults), no buffers
#   c100 = LEAF_COEF=100 INTERNAL_COEF=100, no buffers
#   main = /mnt/scratch/enn_bench/main (3246c4e; leaf 128 lnN, no internal cap)
# Appends to OUTDIR/{c30,c100,main}.jsonl. Compare with paired_quality.py-style stats (fig_0247.py).
O=$(realpath $1); D=$2; N=$3; shift 3; R=$(realpath evals/meta_loop/quality_cmp.py); M=${MAIN:-/mnt/scratch/enn_bench/main}
run() { W=/home/dsweet/bench_work/q47_$1_$S; rm -rf $W; (cd $2 && env $3 PYTHONPATH=src systemd-run --user --scope -q -p MemoryMax=1G -p MemoryHigh=1G -p CPUQuota=100% python $R $D $N $S $W $O/$1.jsonl >> $O/$1.log 2>&1); rm -rf $W; }
for S in "$@"; do
  run c30 . "ENN_PBUF=0 LEAF_COEF=30 INTERNAL_COEF=10"
  run c100 . "ENN_PBUF=0 LEAF_COEF=100 INTERNAL_COEF=100"
  run main $M "ENN_PBUF=0"
done
echo DONE >> $O/budget.log
