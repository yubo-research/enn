#!/bin/bash
# report 0250 third chain: main (3246c4e) under the 1 GB cap, adds + queries to 3e6 (from repo root)
until grep -q CHAIN2_DONE reports/report-0250-data/chain.log 2>/dev/null; do sleep 20; done
PP=/mnt/scratch/enn_bench/main/src REP=_main evals/meta_loop/capq_0250.sh reports/report-0250-data/cap1G 100 3e6 1G 3000 0
echo CHAIN3_DONE $(date +%T) >> reports/report-0250-data/chain.log
