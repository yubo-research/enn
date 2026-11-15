#!/bin/bash
# Usage: advise_ab.sh D N SEED OUTFILE [CAP] -- A/B of the insert-group madvise (ENN_NO_ADVISE) on add time under a cgroup cap.
# Prints add_us_row (quality_cmp.py) with and without the advise. Store on real disk (/home/dsweet/bench_work).
D=$1; N=$2; S=$3; O=$4; CAP=${5:-1G}
R=$(cd "$(dirname "$0")" && pwd)/quality_cmp.py
for MODE in advise noadvise; do
  W=/home/dsweet/bench_work/ab_${MODE}_$S; rm -rf $W
  E=""; [ $MODE = noadvise ] && E="ENN_ADVISE=off"
  echo "mode=$MODE D=$D N=$N cap=$CAP" >>$O
  env $E PYTHONPATH=src systemd-run --user --scope -q -p MemoryMax=$CAP -p MemoryHigh=$CAP -p CPUQuota=100% \
    python $R $D $N $S $W /tmp/ab_$$.jsonl 2>&1 | tail -1 | cut -c1-200 >>$O
  rm -rf $W
done
