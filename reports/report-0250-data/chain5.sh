#!/bin/bash
# report 0250: re-rank pool size vs query reads (code_extra 30 vs 0), 1G cap, no buffers
until grep -q CHAIN4_DONE reports/report-0250-data/chain.log 2>/dev/null; do sleep 20; done
A=reports/report-0250-data/extra; mkdir -p $A
for N in 1e5 1e6 3e6; do
  W=/home/dsweet/bench_work/qextra; rm -rf $W
  timeout 3600 systemd-run --user --scope -q -p MemoryMax=1G -p MemoryHigh=1G -p CPUQuota=100% \
    -E NQ=100 -E PYTHONPATH=/home/dsweet/bench_work/so_pread_0250:evals/meta_loop \
    python evals/meta_loop/qattr_0250.py 100 $N $W $A/extra_$N.json nvme0n1p1 base extra0 base extra0 > $A/extra_$N.log 2>&1
  rm -rf $W
done
echo CHAIN5_DONE $(date +%T) >> reports/report-0250-data/chain.log
