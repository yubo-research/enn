#!/bin/bash
# Usage: ann_capq.sh OUTDIR D MAXN CAP SECONDS NAME   (report-ann; capped add+query sweep with ablation knobs)
# Env: ABL="leaf_coef=30,code_extra=0,..." (setters applied before the sweep), NQ/NF (default 100/10, 30/5 at 1G), REP (file suffix).
# Store on EBS under /home/dsweet/bench_work; never run another job beside a capped sweep (0251).
O=$1; D=$2; MAXN=$3; C=$4; S=$5; NAME=$6; mkdir -p $O; O=$(realpath $O)
H=$(cd "$(dirname "$0")" && pwd)/ann_capq.py
PP=$(cd "$(dirname "$0")/../.." && pwd)/src
W=/home/dsweet/bench_work/ann_capq_$NAME; rm -rf $W; F=$O/capq_D${D}_${C}_$NAME${REP}.jsonl; rm -f $F
timeout $S systemd-run --user --scope -q -p MemoryMax=$C -p MemoryHigh=$C -p CPUQuota=100% \
  -E FINCORE=1 -E ABL="$ABL" \
  -E NQ=${NQ:-$([[ $C == 1G ]] && echo 30 || echo 100)} -E NF=${NF:-$([[ $C == 1G ]] && echo 5 || echo 10)} \
  -E PYTHONPATH=$PP python $H $D $MAXN $W $F nvme0n1p1 > ${F%.jsonl}.log 2>&1
echo STORE $NAME$REP $(du -sb $W | cut -f1) bytes >> $O/sweep.log; rm -rf $W
echo DONE $NAME$REP $(date +%T) >> $O/sweep.log
