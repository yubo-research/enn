#!/bin/bash
# Usage: leafcap_quality_0254.sh OUT D N SEEDS "CAP1 CAP2 ..."   (report 0254)
# Paired recall/NRMSE/loglik/timing (recall_cmp.py) per leaf-capacity build. Overlay for cap C must exist at
# $OV/ov_cC (build: [ENN_LEAF_CAP=C] evals/meta_loop/overlay_0252.sh ov_cC; default OV=/home/dsweet/bench_work).
# 2G scope, 1 CPU, arms alternate within each seed. env: INTRINSIC, FIRST (first seed), QW (work-dir base).
# Writes OUT/c<C>.jsonl; compare with pair_0250.py OUT/c256.jsonl OUT/c64.jsonl (main: report-0246-data/score/main.jsonl at D=10).
O=$1; D=$2; N=$3; S=$4; ARMS=$5; mkdir -p $O; H=$(cd "$(dirname "$0")" && pwd); OV=${OV:-/home/dsweet/bench_work}
W=${QW:-/mnt/scratch}/lq_$$
for s in $(seq ${FIRST:-0} $((${FIRST:-0}+S-1))); do for c in $ARMS; do
  rm -rf $W
  timeout 3000 systemd-run --user --scope -q -p MemoryMax=2G -p MemoryHigh=2G -p CPUQuota=100% \
    -E INTRINSIC=${INTRINSIC:-0} -E PYTHONPATH=$OV/ov_c$c python $H/recall_cmp.py $D $N $s 0 $W $O/c$c.jsonl > /dev/null 2>> $O/err.log
  rm -rf $W
done; done
echo DONE >> $O/err.log
