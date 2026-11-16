#!/bin/bash
# report 0251: clean reruns (nothing else running) after the P30 sweep; run from repo root
while pgrep -f "capq_D100_256M_pb0_P30" > /dev/null; do sleep 20; done
S=evals/meta_loop/capq_0250.sh; O=reports/report-0251-data/cap256
export PP=/home/dsweet/bench_work/pp_0251
PATIENCE=10 REP=_P10r $S $O 100 3e6 256M 3600 0
PATIENCE=0 REP=_P0r $S $O 100 3e6 256M 3600 0
echo CAP2_DONE $(date +%T) >> $O/sweep.log
