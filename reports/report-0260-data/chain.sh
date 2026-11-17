#!/bin/bash
# report 0260 run chain (from repo root); everything sequential
cd /home/dsweet/projects/enn
D=reports/report-0260-data
bash evals/meta_loop/capq_0260.sh $D/cap256 10 3e6 256M 3600 "g10:10:1:2:10:ov_0260h def:0:0:0:0:ov_0260h g30:10:1:2:30:ov_0260h main:0:0:0:0:ov_0260main"
echo CAP_DONE $(date +%T) >> $D/chain.log
for n in 1e5 1e6; do
  bash evals/meta_loop/quality_0259.sh $D/qual/d10_$n 10 $n 10 "0:0:0:0:0:ov_0260h 10:10:1:2:10:ov_0260h 30:10:1:2:30:ov_0260h"
  echo QUAL_$n $(date +%T) >> $D/chain.log
done
echo CHAIN_DONE $(date +%T) >> $D/chain.log
