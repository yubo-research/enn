#!/bin/bash
# Usage: code4_cap.sh OUTDIR D MAXN CAP SECONDS ARM...   (run from repo root; report 0241)
# Capped add sweep (bytes_per_add.py, FINCORE residency per file every ~30 s) per page-code arm, sequential,
# 1 CPU, store on the root disk. Arm = ENN_CODE4 value (0 8-bit, 1 raw 4-bit, 2 residual 4-bit);
# suffix i sets ENN_RESID_INHERIT=1, tX sets ENN_RESID_TAU=X (report 0239). PP = PYTHONPATH (default src).
O=$1; D=$2; MAXN=$3; C=$4; S=$5; shift 5; mkdir -p $O; O=$(realpath $O)
H=$(cd "$(dirname "$0")" && pwd)/bytes_per_add.py
for a in "$@"; do
  c=${a:0:1}; tau=0; inh=0; [[ $a =~ t([0-9.]+) ]] && tau=${BASH_REMATCH[1]}; [[ $a == *i* ]] && inh=1
  W=/home/dsweet/bench_work/c4cap_$a; rm -rf $W; F=$O/bpa_D${D}_${C}_c$a.jsonl; rm -f $F
  timeout $S systemd-run --user --scope -q -p MemoryMax=$C -p MemoryHigh=$C -p CPUQuota=100% \
    -E ENN_CODE4=$c -E ENN_RESID_TAU=$tau -E ENN_RESID_INHERIT=$inh -E FINCORE=1 -E PYTHONPATH=${PP:-src} \
    python $H $D $MAXN $W $F nvme0n1p1 > ${F%.jsonl}.log 2>&1
  rm -rf $W
  echo DONE $a $(date +%T) >> $O/sweep.log
done
