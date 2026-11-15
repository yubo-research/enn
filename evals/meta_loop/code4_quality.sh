#!/bin/bash
# Usage: code4_quality.sh OUTDIR D "NS" SEEDS PP   (repo root)   report 0237
# Prediction quality (recall_cmp.py: recall@10, NRMSE, loglik) for ENN_CODE4=0 and 1, run in parallel
# per seed on local NVMe (env INTRINSIC passes through to recall_cmp.py) (/mnt/scratch); quality does not depend on timing. Compare: paired_quality.py on1 off.
O=$(realpath $1); D=$2; NS=$3; S=$4; P=$5; H=$(cd "$(dirname "$0")" && pwd)/recall_cmp.py; mkdir -p $O
for N in $NS; do for s in $(seq 1 $S); do
  for a in 0 1; do
    W=/mnt/scratch/c4q_$a; rm -rf $W
    (cd /tmp && ENN_CODE4=$a PYTHONPATH=$P timeout 1800 python $H $D $N $s 30 $W $O/q_D${D}_c4$a.jsonl > /dev/null 2>&1; rm -rf $W) &
  done; wait
done; done
