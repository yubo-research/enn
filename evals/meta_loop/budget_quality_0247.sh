#!/bin/bash
# Usage: budget_quality_0247.sh OUTDIR LANE   (repo root; report 0247; run LANE=0 and LANE=1 in parallel)
# Recall@10 + quality at D=10 N=1e6 (recall_cmp.py, uncapped on /mnt/scratch: quality does not depend on timing).
# Arms: B = leaf 30 / internal 30; C = leaf 100 / internal 100; D = C with every-level buffers. Seeds LANE, LANE+2, ..., 9.
# Paired against main/head0/pbuf of reports/report-0246-data/score with fig_0247.py.
O=$(realpath $1); L=$2; R=$(realpath evals/meta_loop/recall_cmp.py)
run() { W=/mnt/scratch/bq47_$1_$S; rm -rf $W; env $3 INTERNAL_COEF=$4 PYTHONPATH=src python $R 10 1e6 $S $2 $W $O/q_$1.jsonl >> $O/lane$L.log 2>&1; rm -rf $W; }
for S in $(seq $L 2 9); do
  run B 30 "ENN_PBUF=0" 30
  run C 100 "ENN_PBUF=0" 100
  run D 100 "ENN_PBUF=1000 ENN_PBUF_ALL=1 ENN_PBUF_PREFETCH=1 ENN_PBUF_PUNCH=1" 100
done
