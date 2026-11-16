#!/bin/bash
# Usage: recall_sweep_0247.sh OUTDIR [SEED]   (repo root; report 0247) seed-0 recall@10 sweep at D=10 N=1e6.
# Branch: leaf c in 10 30 100 128 (internal 10), buffers off/on -> off.jsonl / pbuf.jsonl; knob probes at c=30
# (STOP_EPS 0 / 0.3, INTERNAL_COEF 30 / 100, RERANK_SKIP 0) -> eps/ic/rr.jsonl; main (3246c4e default) -> main.jsonl.
O=$(realpath $1); S=${2:-0}; R=$(realpath evals/meta_loop/recall_cmp.py); M=${MAIN:-/mnt/scratch/enn_bench/main}
run() { W=/mnt/scratch/rs47_$$; rm -rf $W; (cd $1 && env $2 PYTHONPATH=src python $R 10 1e6 $S $3 $W $O/$4.jsonl >> $O/sweep.log 2>&1); rm -rf $W; }
B="ENN_PBUF=1000 ENN_PBUF_ALL=1 ENN_PBUF_PREFETCH=1 ENN_PBUF_PUNCH=1"
for C in 10 30 100 128; do run . "ENN_PBUF=0 INTERNAL_COEF=10" $C off; run . "$B INTERNAL_COEF=10" $C pbuf; done
for E in 0 0.3; do run . "ENN_PBUF=0 INTERNAL_COEF=10 STOP_EPS=$E" 30 eps; done
for I in 30 100; do run . "ENN_PBUF=0 INTERNAL_COEF=$I" 30 ic; done
run . "ENN_PBUF=0 INTERNAL_COEF=10 RERANK_SKIP=0" 30 rr
run $M "ENN_PBUF=0" 0 main
