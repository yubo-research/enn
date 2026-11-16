#!/bin/bash
# report 0250 quality: exact re-rank pool k+30 (default) vs k (CODE_EXTRA=0), no buffers, src build
export LEAF=0
for D in 100 10; do
  if [ $D = 100 ]; then export INTRINSIC=5; else unset INTRINSIC; fi
  for e in def 0; do
    ( [ $e = 0 ] && export CODE_EXTRA=0; QDIR=/mnt/scratch/q0250e$e ARMS="0" evals/meta_loop/pbuf_quality.sh reports/report-0250-data/qual_extra/e$e $D "1e5 3e5" 5 ) &
  done
  wait
done
echo QEDONE $(date +%T) >> reports/report-0250-data/qual_extra/qsweep.log
