#!/bin/bash
# Usage: resid_quality.sh OUTDIR D "NS" SEEDS PP   (repo root)   report 0238
# Prediction quality (recall_cmp.py: recall@10, NRMSE, loglik) for ENN_CODE4 arms in $ARMS (default "0 2"),
# run in parallel per seed on local NVMe (/mnt/scratch); quality does not depend on timing.
# env INTRINSIC passes through. Compare: python paired_quality.py OUT/q_D{D}_c42.jsonl OUT/q_D{D}_c40.jsonl
O=$(realpath $1); D=$2; NS=$3; S=$4; P=$5; H=$(cd "$(dirname "$0")" && pwd)/recall_cmp.py; mkdir -p $O
read -ra A <<< "${ARMS:-0 2}"
for N in $NS; do for s in $(seq 1 $S); do
  for a in "${A[@]}"; do
    W=/mnt/scratch/rq_$a; rm -rf $W
    (cd /tmp && ENN_CODE4=$a PYTHONPATH=$P timeout 1800 python $H $D $N $s 30 $W $O/q_D${D}_c4$a.jsonl > /dev/null 2>&1; rm -rf $W) &
  done; wait
done; done
