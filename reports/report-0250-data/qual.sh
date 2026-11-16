#!/bin/bash
# report 0250 quality: R=3000 every-level buffers, ENN_PBUF_CHARGE 0 vs 1 (overlay so_charge_0250), plus no-buffer arm
P=/home/dsweet/bench_work/so_charge_0250
export ENN_PBUF_ALL=1 ENN_PBUF_PREFETCH=1 ENN_PBUF_PUNCH=1 LEAF=0
for D in 100 10; do
  if [ $D = 100 ]; then export INTRINSIC=5; else unset INTRINSIC; fi
  for c in 0 1; do
    ENN_PBUF_CHARGE=$c QDIR=/mnt/scratch/q0250c$c PP=$P ARMS="3000" evals/meta_loop/pbuf_quality.sh reports/report-0250-data/qual/c$c $D "1e5 3e5" 5 &
  done
  wait
done
echo QDONE $(date +%T) >> reports/report-0250-data/qual/qsweep.log
