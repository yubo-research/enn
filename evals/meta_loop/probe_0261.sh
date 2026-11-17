#!/bin/bash
# Usage: probe_0261.sh OUT D N "name:lloyd_every:lloyd_graph:walk_graph:leaf_coef:pbuf:lloyd_flush ..."   (report 0261)
# In-RAM (1G cap, 1 CPU) add time + per-query leaves, scanned rows (leaf + buffer) and time, one arm after another.
O=$1; D=$2; N=$3; ARMS=$4; H=$(cd "$(dirname "$0")" && pwd)/probe_0261.py
for a in $ARMS; do
  IFS=: read name lloyd graph walk leaf pbuf flush <<< "$a"
  on=$([[ $pbuf == 0 ]] && echo 0 || echo 1)
  W=/home/dsweet/bench_work/probe0261_$name; rm -rf $W
  systemd-run --user --scope -q -p MemoryMax=1G -p MemoryHigh=1G -p CPUQuota=100% \
    -E ENN_PBUF=$pbuf -E ENN_LOG_PACE=1 -E ENN_PBUF_ALL=$on -E ENN_PBUF_PREFETCH=$on -E ENN_PBUF_PUNCH=$on \
    -E ENN_LLOYD_EVERY=$lloyd -E ENN_LLOYD_GRAPH=$graph -E ENN_WALK_GRAPH=$walk -E LEAF_COEF=$leaf -E ENN_LLOYD_FLUSH=$flush ${XV:+-E $XV} \
    -E ARM=$name -E NQ=${NQ:-300} -E PYTHONPATH=/home/dsweet/bench_work/${OV:-ov_0261} python $H $D $N $W $O
  rm -rf $W
done
