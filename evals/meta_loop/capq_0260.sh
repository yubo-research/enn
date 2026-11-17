#!/bin/bash
# Usage: capq_0260.sh OUTDIR D MAXN CAP SECONDS "name:lloyd_every:lloyd_graph:walk_graph:leaf_coef:overlay ..."   (report 0260)
# Capped add + query sweep (capq_0260.py -> capq_0250.py) per arm, one systemd-run scope each (MemoryMax=MemoryHigh=CAP,
# CPUQuota=100%), store on the root EBS disk (/home/dsweet/bench_work). overlay = dir under /home/dsweet/bench_work.
# leaf_coef 0 = default (100 lnN). Arms run one after another, never concurrently. NQ/NF as capq_0250.sh.
O=$1; D=$2; MAXN=$3; C=$4; S=$5; ARMS=$6; mkdir -p $O; O=$(realpath $O)
H=$(cd "$(dirname "$0")" && pwd)/capq_0260.py
for a in $ARMS; do
  IFS=: read name lloyd graph walk leaf ov <<< "$a"
  W=/home/dsweet/bench_work/capq0260_$name; rm -rf $W; F=$O/capq_D${D}_${C}_$name${REP}.jsonl; rm -f $F
  timeout $S systemd-run --user --scope -q -p MemoryMax=$C -p MemoryHigh=$C -p CPUQuota=100% \
    -E ENN_PBUF=0 -E ENN_LOG_PACE=1 -E ENN_PBUF_ALL=0 -E ENN_PBUF_PREFETCH=0 -E ENN_PBUF_PUNCH=0 -E FINCORE=1 \
    -E ENN_LLOYD_EVERY=$lloyd -E ENN_LLOYD_GRAPH=$graph -E ENN_WALK_GRAPH=$walk -E LEAF_COEF=$leaf \
    -E NQ=${NQ:-$([[ $C == 1G ]] && echo 30 || echo 100)} -E NF=${NF:-$([[ $C == 1G ]] && echo 5 || echo 10)} \
    -E PYTHONPATH=/home/dsweet/bench_work/$ov \
    python $H $D $MAXN $W $F nvme0n1p1 > ${F%.jsonl}.log 2>&1
  rm -rf $W
  echo DONE $name$REP $(date +%T) >> $O/sweep.log
done
