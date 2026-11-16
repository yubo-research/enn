#!/bin/bash
# report 0251 quality lane (NVMe work dirs), run from repo root
S=evals/meta_loop/patience_quality_0251.sh; Q=reports/report-0251-data/qual
export PP=/home/dsweet/bench_work/pp_0251
INTRINSIC=5 $S $Q/d100i5_1e5 100 1e5 10 "main"
$S $Q/d10_1e5 10 1e5 10 "30"
FIRST=5 $S $Q/d10_1e6 10 1e6 5 "0 10"
$S $Q/d10_1e6 10 1e6 10 "30"
echo QCHAIN_DONE $(date +%T) >> $Q/chain.log
