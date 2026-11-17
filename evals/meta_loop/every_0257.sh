#!/bin/bash
O=$1; shift; mkdir -p $O; O=$(realpath $O)
H=$(cd "$(dirname "$0")" && pwd)/growth_0256.py
B=/home/dsweet/bench_work/g0257
PP=${PP:-/home/dsweet/bench_work/ov_c64}
for spec in "$@"; do
  IFS=: read arm n seed m full every lloyd <<< "$spec"
  W=$B/w_${arm}_${n}_${seed}; rm -rf $W
  PYTHONPATH=$PP ARM=$arm CAP=64 OFFLINE= LLOYD=0 NQ=${NQ:-300} ENN_REASSIGN=$m ENN_REASSIGN_FULL=$full \
    ENN_REASSIGN_EVERY=$every ENN_LLOYD_EVERY=${lloyd:-0} python $H ${D:-10} $n $seed $W $O/every_${arm}.jsonl > $O/every_${arm}_${n}_${seed}.log 2>&1
  rm -rf $W
  echo "DONE $arm $n $seed $(date +%T)" >> $O/sweep.log
done
echo ALLDONE $(date +%T) >> $O/sweep.log
