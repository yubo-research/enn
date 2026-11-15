#!/bin/bash
# Usage: code4_ram_ab.sh OUT D N REPS BASE_PP HEAD_PP   (repo root; 1G cap, 1 CPU)   report 0237
# Paired in-RAM add/query timing (advise_data_ab.py) of three arms, order rotated per seed:
# base = BASE_PP build, ENN_CODE4=0; off = HEAD_PP, ENN_CODE4=0; on = HEAD_PP, ENN_CODE4=1. Arm label in "advise".
O=$1; D=$2; N=$3; R=$4; B=$5; HP=$6; H=$(cd "$(dirname "$0")" && pwd)/advise_data_ab.py
ARMS=(base off on)
for s in $(seq 1 $R); do for j in 0 1 2; do
  a=${ARMS[$(( (s + j) % 3 ))]}; P=$HP; C=0
  [ $a = base ] && P=$B; [ $a = on ] && C=1
  W=/home/dsweet/bench_work/c4ram; rm -rf $W
  (cd /tmp && timeout 900 systemd-run --user --scope -q -p MemoryMax=1G -p MemoryHigh=1G -p CPUQuota=100% \
    -E ENN_DATA_ADVISE=$a -E ENN_CODE4=$C -E SEED=$s -E PYTHONPATH=$P python $H $D $N $W $O)
  rm -rf $W
done; done
