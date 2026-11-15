#!/bin/bash
# Usage: add_gate_cmp.sh LABEL CHECKOUT OUTDIR [SEEDS] -- add/query/RSS/quality for N=1e3..1e5, D=10,100, 1 GB cgroup cap.
# Same seeds for every checkout; results appended to OUTDIR/LABEL.jsonl. Used by report 0232.
L=$1; C=$(realpath $2); O=$3; NS=${4:-5}; mkdir -p $O; O=$(realpath $O)
R=$(cd "$(dirname "$0")" && pwd)/quality_cmp.py
for N in 1e3 1e4 1e5; do for D in 10 100; do for S in $(seq 0 $((NS-1))); do
  W=/home/dsweet/bench_work/ag_${L}_${D}_${N}_${S}; rm -rf $W
  (cd $C && PYTHONPATH=src systemd-run --user --scope -q -p MemoryMax=1G -p MemoryHigh=1G -p CPUQuota=100% \
    python $R $D $N $S $W $O/${L}.jsonl >>$O/${L}.log 2>&1)
  rm -rf $W
done; done; done
echo DONE >>$O/${L}.log
