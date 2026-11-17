#!/bin/bash
# Usage: capq_0266.sh OUTDIR D MAXN CAP SECONDS NAME   (report 0266; the library has no env flags after 0266)
# Capped add + query sweep of the current build (capq_0260.py -> capq_0250.py) in one systemd-run scope (MemoryMax=MemoryHigh=CAP, 1 CPU).
# Env: OV = wheel overlay under /home/dsweet/bench_work (default: repo src), LEAF_COEF (default library budget), NQ/NF (default 100/10, 30/5 at 1G), REP (file suffix).
# Store on EBS under /home/dsweet/bench_work; never run another job beside a capped sweep (0251).
O=$1; D=$2; MAXN=$3; C=$4; S=$5; NAME=$6; mkdir -p $O; O=$(realpath $O)
H=$(cd "$(dirname "$0")" && pwd)/capq_0260.py
PP=${OV:+/home/dsweet/bench_work/$OV}; PP=${PP:-$(cd "$(dirname "$0")/../.." && pwd)/src}
W=/home/dsweet/bench_work/capq0266_$NAME; rm -rf $W; F=$O/capq_D${D}_${C}_$NAME${REP}.jsonl; rm -f $F
timeout $S systemd-run --user --scope -q -p MemoryMax=$C -p MemoryHigh=$C -p CPUQuota=100% \
  -E FINCORE=1 ${LEAF_COEF:+-E LEAF_COEF=$LEAF_COEF} \
  -E NQ=${NQ:-$([[ $C == 1G ]] && echo 30 || echo 100)} -E NF=${NF:-$([[ $C == 1G ]] && echo 5 || echo 10)} \
  -E PYTHONPATH=$PP python $H $D $MAXN $W $F nvme0n1p1 > ${F%.jsonl}.log 2>&1
echo STORE $NAME$REP $(du -sb $W | cut -f1) bytes >> $O/sweep.log; rm -rf $W
echo DONE $NAME$REP $(date +%T) >> $O/sweep.log
