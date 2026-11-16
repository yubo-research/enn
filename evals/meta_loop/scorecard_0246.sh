#!/bin/bash
# Usage: scorecard_0246.sh OUTDIR   (repo root; report 0246)
# Paired five-metric scorecard (scorecard_0226.sh: add us/row, query s, peak RSS, NRMSE, loglik; 1 GB cap,
# 1 CPU, D in {10,100}, N in 1e3..1e6, 10 seeds (5 at 1e6)) for HEAD with every-level buffers
# (ENN_PBUF=1000 ENN_PBUF_ALL=1 PREFETCH=1 PUNCH=1, label pbuf) and main (/mnt/scratch/enn_bench/main, 3246c4e).
# Stats/figure: python evals/meta_loop/scorecard_0226.py-style via scorecard_0246.py OUTDIR.
O=$1; H=$(cd "$(dirname "$0")" && pwd)
ENN_PBUF=1000 ENN_PBUF_ALL=1 ENN_PBUF_PREFETCH=1 ENN_PBUF_PUNCH=1 $H/scorecard_0226.sh pbuf . $O
env -u ENN_PBUF -u ENN_PBUF_ALL $H/scorecard_0226.sh main ${MAIN:-/mnt/scratch/enn_bench/main} $O
