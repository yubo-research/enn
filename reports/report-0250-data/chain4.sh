#!/bin/bash
until grep -q CHAIN3_DONE reports/report-0250-data/chain.log 2>/dev/null; do sleep 20; done
evals/meta_loop/pread_ab_0250.sh reports/report-0250-data/pread /home/dsweet/bench_work/so_pread_0250 3 1e5 1e6 3e6
echo CHAIN4_DONE $(date +%T) >> reports/report-0250-data/chain.log
