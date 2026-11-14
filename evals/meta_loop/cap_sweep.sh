#!/bin/bash
# Usage: cap_sweep.sh OUTDIR D MAXN CAP:SECONDS ...   e.g.  cap_sweep.sh out 100 1e7 64M:600 256M:900 1G:1800
# Runs bytes_per_add.py under each cgroup memory cap (1 CPU), sequentially, store on the root disk.
O=$1; D=$2; MAXN=$3; shift 3; mkdir -p $O; O=$(realpath $O)
H=$(cd "$(dirname "$0")" && pwd)/bytes_per_add.py
for cs in "$@"; do
  C=${cs%%:*}; S=${cs##*:}; W=/home/dsweet/bench_work/bpa_$C; rm -rf $W; rm -f $O/bpa_D${D}_$C.jsonl
  PYTHONPATH=src timeout $S systemd-run --user --scope -q -p MemoryMax=$C -p MemoryHigh=$C -p CPUQuota=100% \
    python $H $D $MAXN $W $O/bpa_D${D}_$C.jsonl nvme0n1p1 > $O/bpa_D${D}_$C.log 2>&1
  rm -rf $W
done
echo DONE >> $O/sweep.log
