#!/bin/bash
# report 0251 capped add+query sweeps (EBS store), run from repo root; waits for EBS quality runs
while pgrep -f "bench_work/pq_" > /dev/null; do sleep 30; done
S=evals/meta_loop/capq_0250.sh; O=reports/report-0251-data/cap256
export PP=/home/dsweet/bench_work/pp_0251
for p in 10 0 30; do PATIENCE=$p REP=_P$p $S $O 100 3e6 256M 3600 0; done
echo CAPCHAIN_DONE $(date +%T) >> $O/sweep.log
