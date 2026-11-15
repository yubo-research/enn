#!/bin/bash
# Usage: resid_ab.sh OUTDIR D MAXN CAP SECONDS REPS   (run from repo root)   report 0238
# Cold+warm query per N for page-code arms ENN_CODE4 in $ARMS (default "0 1 2": 8-bit, raw 4-bit,
# residual 4-bit vs 8-bit leaf anchor), one process per arm, arm order rotated per rep.
# Arm "2t0.1" sets ENN_RESID_TAU=0.1, "2i" ENN_RESID_INHERIT=1 (report 0239). Arm suffix n (e.g. 2n) sets ENN_REFINE_MORE=0. LD_LIBRARY_PATH passes through (wheel builds).
# PP = PYTHONPATH of the build (default src); INTRINSIC=m for data near an m-dim subspace (code4_ab.py).
O=$1; D=$2; N=$3; C=$4; S=$5; R=${6:-2}; mkdir -p $O; O=$(realpath $O)
H=$(cd "$(dirname "$0")" && pwd)/code4_ab.py
read -ra A <<< "${ARMS:-0 1 2}"
for r in $(seq 1 $R); do
  k=$(( (r - 1) % ${#A[@]} )); ORDER=("${A[@]:k}" "${A[@]:0:k}")
  for a in "${ORDER[@]}"; do
    b=${a%n}; rm=1; [ "$b" != "$a" ] && rm=0; c=${b:0:1}; tau=0; inh=0; [[ $b =~ t([0-9.]+) ]] && tau=${BASH_REMATCH[1]}; [[ $b == *i* ]] && inh=1
    W=/home/dsweet/bench_work/c4_${D}; rm -rf $W; F=$O/c4_D${D}_${C}_a${a}_r${r}.jsonl; rm -f $F
    PYTHONPATH=${PP:-src} timeout $S systemd-run --user --scope -q -p MemoryMax=$C -p MemoryHigh=$C -p CPUQuota=100% \
      -E ENN_CODE4=$c -E ENN_REFINE_MORE=$rm -E ENN_RESID_TAU=$tau -E ENN_RESID_INHERIT=$inh -E INTRINSIC=${INTRINSIC:-0} python $H c4_$a $D $N $W $F nvme0n1p1 > ${F%.jsonl}.log 2>&1
    rm -rf $W
  done
done
echo DONE $D $N $C >> $O/sweep.log
