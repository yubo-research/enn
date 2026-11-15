#!/bin/bash
# Usage: scorecard_0226.sh LABEL CHECKOUT OUTDIR -- paired five-metric run (add us/row, query s, RSS, NRMSE, loglik)
# under a 1 GB cgroup cap, store on a real disk. Same seeds for both checkouts. Uses quality_cmp.py.
L=$1; C=$(realpath $2); O=$3; mkdir -p $O; O=$(realpath $O)
R=$(cd "$(dirname "$0")" && pwd)/quality_cmp.py
for N in 1e3 1e4 1e5 1e6; do for D in 10 100; do
  NS=10; [ $N = 1e6 ] && NS=5
  for S in $(seq 0 $((NS-1))); do
    W=/home/dsweet/bench_work/sc_${L}_${D}_${N}_${S}; rm -rf $W
    (cd $C && PYTHONPATH=src systemd-run --user --scope -q -p MemoryMax=1G -p MemoryHigh=1G -p CPUQuota=100% \
      python $R $D $N $S $W $O/${L}.jsonl >>$O/${L}.log 2>&1)
    rm -rf $W
  done
done; done
echo DONE >>$O/${L}.log
