#!/bin/bash
# Usage: resid_ram_ab.sh OUT D N REPS PP   (repo root; 1G cap, 1 CPU)   report 0238
# Paired in-RAM add/query timing (advise_data_ab.py, uniform data) of page-code arms in $ARMS (default "0 1 2")
# on one build, order rotated per seed; arm label "c4_<arm>" in field "advise". Infix t (e.g. 2t0.1) sets ENN_RESID_TAU, i sets ENN_RESID_INHERIT=1 (0239). Suffix n (e.g. 2n) sets
# ENN_REFINE_MORE=0. LD_LIBRARY_PATH passes through (wheel builds). Stats: resid_ram_stats.py OUT
O=$1; D=$2; N=$3; R=$4; P=${5:-src}; H=$(cd "$(dirname "$0")" && pwd)/advise_data_ab.py
read -ra A <<< "${ARMS:-0 1 2}"
for s in $(seq 1 $R); do for j in $(seq 0 $((${#A[@]} - 1))); do
  a=${A[$(( (s + j) % ${#A[@]} ))]}; b=${a%n}; rm=1; [ "$b" != "$a" ] && rm=0; c=${b:0:1}; tau=0; inh=0; [[ $b =~ t([0-9.]+) ]] && tau=${BASH_REMATCH[1]}; [[ $b == *i* ]] && inh=1
  W=/home/dsweet/bench_work/rram; rm -rf $W
  (cd /tmp && timeout 900 systemd-run --user --scope -q -p MemoryMax=1G -p MemoryHigh=1G -p CPUQuota=100% \
    -E ENN_DATA_ADVISE=c4_$a -E ENN_CODE4=$c -E ENN_REFINE_MORE=$rm -E ENN_RESID_TAU=$tau -E ENN_RESID_INHERIT=$inh -E SEED=$s -E PYTHONPATH=$P python $H $D $N $W $O)
  rm -rf $W
done; done
