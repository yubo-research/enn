#!/bin/bash
# Usage: patience_quality_0251.sh OUT D N SEEDS "P1 P2 ..."   (report 0251)
# Paired recall/NRMSE/loglik (recall_cmp.py) per patience coefficient P (0 = off = default), 2G scope, 1 CPU.
# Arm "main" runs the main worktree (MAIN_PP, default /mnt/scratch/enn_bench/main/src) without PATIENCE.
# env: PP overlay for patience arms; INTRINSIC; FIRST (first seed, default 0); QW work-dir base (default /mnt/scratch).
# Writes OUT/p<P>.jsonl; compare with pair_0250.py OUT/p<P>.jsonl OUT/p0.jsonl
O=$1; D=$2; N=$3; S=$4; ARMS=$5; mkdir -p $O; H=$(cd "$(dirname "$0")" && pwd)
W=${QW:-/mnt/scratch}/pq_$$
for s in $(seq ${FIRST:-0} $((${FIRST:-0}+S-1))); do for p in $ARMS; do
  rm -rf $W
  if [ "$p" = main ]; then E="-E PYTHONPATH=${MAIN_PP:-/mnt/scratch/enn_bench/main/src}"
  else E="-E PATIENCE=$p -E PYTHONPATH=${PP:-/home/dsweet/projects/enn/src}"; fi
  timeout 3000 systemd-run --user --scope -q -p MemoryMax=2G -p MemoryHigh=2G -p CPUQuota=100% \
    -E INTRINSIC=${INTRINSIC:-0} $E python $H/recall_cmp.py $D $N $s 0 $W $O/p$p.jsonl > /dev/null 2>> $O/err.log
  rm -rf $W
done; done
echo DONE >> $O/err.log
