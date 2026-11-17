#!/bin/bash
# report 0260 run chain 2 (after chain.sh's def/g10/main arms); sequential
cd /home/dsweet/projects/enn
D=reports/report-0260-data
bash evals/meta_loop/capq_0260.sh $D/cap256 10 3e6 256M 3600 "g10w:10:1:2:10:ov_0260w"
echo CAP2_DONE $(date +%T) >> $D/chain.log
bash evals/meta_loop/quality_0259.sh $D/qual/d10_1e5 10 1e5 10 "0:0:0:0:0:ov_0260h 10:10:1:2:10:ov_0260h 30:10:1:2:30:ov_0260h 110:10:1:2:10:ov_0260w"
echo QUAL_1e5 $(date +%T) >> $D/chain.log
bash evals/meta_loop/quality_0259.sh $D/qual/d10_1e6 10 1e6 10 "10:10:1:2:10:ov_0260h 30:10:1:2:30:ov_0260h"
echo QUAL_1e6 $(date +%T) >> $D/chain.log
echo CHAIN_DONE $(date +%T) >> $D/chain.log
