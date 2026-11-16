#!/bin/bash
# report 0250 second chain: waits for chain.sh, then R=1000 capped arm and scan attribution (from repo root)
until grep -q CHAIN_DONE reports/report-0250-data/chain.log 2>/dev/null; do sleep 20; done
until grep -q QDONE reports/report-0250-data/qual/qsweep.log 2>/dev/null; do sleep 20; done
evals/meta_loop/capq_0250.sh reports/report-0250-data/cap256 100 3e6 256M 3600 1000aph
A=reports/report-0250-data/attr; mkdir -p $A
P=/home/dsweet/bench_work/so_charge_0250
for N in 1e6 3e6; do for arm in 0 3000 3000c; do
  W=/home/dsweet/bench_work/qattr_$arm; rm -rf $W
  pb=${arm%c}; ch=$([[ $arm == *c ]] && echo 1 || echo 0)
  timeout 3600 systemd-run --user --scope -q -p MemoryMax=1G -p MemoryHigh=1G -p CPUQuota=100% \
    -E ENN_PBUF=$pb -E ENN_PBUF_ALL=1 -E ENN_PBUF_PREFETCH=1 -E ENN_PBUF_PUNCH=1 -E ENN_PBUF_CHARGE=$ch -E NQ=100 \
    -E PYTHONPATH=$P:evals/meta_loop python evals/meta_loop/qattr_0250.py 100 $N $W $A/qattr_${arm}_$N.json nvme0n1p1 base again \
    > $A/qattr_${arm}_$N.log 2>&1
  rm -rf $W
done; done
echo CHAIN2_DONE $(date +%T) >> reports/report-0250-data/chain.log
