#!/bin/bash
# Usage: stop_trace_0251.sh OUT   (report 0251) walk-trace stop-rule grid, in RAM, seed 0
# env NQ (default 200); SPECS="D:N:INTRINSIC ..." (default grid below); RADIUS_W passes through
O=$1; mkdir -p $O; H=$(cd "$(dirname "$0")" && pwd); W=/home/dsweet/bench_work/st_w${TAG}
for spec in ${SPECS:-10:1e4:0 10:1e5:0 10:1e6:0 100:1e4:5 100:1e5:5 100:3e5:5 100:1e4:0 100:1e5:0}; do
  IFS=: read -r d n i <<< "$spec"; rm -rf $W
  INTRINSIC=$i PYTHONPATH=${PP:-/home/dsweet/projects/enn/src} timeout 3000 python $H/stop_trace_0251.py $d $n 0 $W $O/trace.jsonl >> $O/summary.txt 2>&1
  rm -rf $W
done
echo DONE >> $O/summary.txt
