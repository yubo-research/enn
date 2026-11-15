#!/bin/bash
# Usage: pbuf_quality.sh OUTDIR D "NS" SEEDS   (repo root; report 0242)
# PP = PYTHONPATH (default src). Prediction quality (recall_cmp.py: recall@10, NRMSE, loglik) per ENN_PBUF arm in $ARMS (default "0 1000"),
# env SEED_FROM (default 1) starts the seed range at SEED_FROM..SEEDS. Arms in parallel per seed on local NVMe (/mnt/scratch); quality does not depend on timing. env INTRINSIC, LEAF
# (default 30 = library default) pass through. Compare: python paired_quality.py OUT/q_D{D}_pb1000.jsonl OUT/q_D{D}_pb0.jsonl
O=$(realpath $1); D=$2; NS=$3; S=$4; H=$(cd "$(dirname "$0")" && pwd)/recall_cmp.py; mkdir -p $O
P=$(realpath ${PP:-src}); read -ra A <<< "${ARMS:-0 1000}"
for N in $NS; do for s in $(seq ${SEED_FROM:-1} $S); do
  for a in "${A[@]}"; do
    W=/mnt/scratch/pbq_$a; rm -rf $W
    (cd /tmp && ENN_PBUF=$a PYTHONPATH=$P timeout 1800 python $H $D $N $s ${LEAF:-30} $W $O/q_D${D}_pb$a.jsonl > /dev/null 2>&1; rm -rf $W) &
  done; wait
done; done
echo DONE $D >> $O/qsweep.log
