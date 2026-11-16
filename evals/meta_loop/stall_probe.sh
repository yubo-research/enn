#!/bin/bash
# Usage: stall_probe.sh OUTDIR D MAXN CAP ARM...   (repo root; report 0244)
# stall_probe.py per arm under a cgroup cap, 1 CPU, store on EBS (/home/dsweet/bench_work, nvme0n1).
# Arm = ENN_PBUF rows + suffixes a (ENN_PBUF_ALL=1), x (ENN_LEAF_EXTENT=1; needs 0244 leaf_extent.patch), p (ENN_PBUF_PREFETCH=1; else 0), h (ENN_PBUF_PUNCH=1, 0245); e.g. 1000a 1000ap 1000aph. PP = PYTHONPATH; FAULTS=1 sets ENN_PBUF_FAULTS=1; TIMERS=1 sets ENN_ADD_TIMERS=1 (0248 phase timers); LOG_LINEAR=1 sets ENN_LOG_GROW_LINEAR=1 (pre-0248 train_x growth); LOG_PACE=0 sets ENN_LOG_PACE=0 (pre-0249 log writeback; default 1).
O=$1; D=$2; MAXN=$3; C=$4; shift 4; mkdir -p $O; O=$(realpath $O); H=$(cd "$(dirname "$0")" && pwd)/stall_probe.py
for a in "$@"; do
  W=/home/dsweet/bench_work/stall_$a; rm -rf $W; F=$O/stall_D${D}_${C}_pb$a.jsonl; rm -f $F
  timeout ${SECS:-3600} systemd-run --user --scope -q -p MemoryMax=$C -p MemoryHigh=$C -p CPUQuota=100% \
    -E ENN_PBUF=${a%%[asxph]*} -E ENN_PBUF_ALL=$([[ $a == *a* ]] && echo 1 || echo 0) -E ENN_LEAF_EXTENT=$([[ $a == *x* ]] && echo 1 || echo 0) -E ENN_PBUF_PREFETCH=$([[ $a == *p* ]] && echo 1 || echo 0) -E ENN_PBUF_PUNCH=$([[ $a == *h* ]] && echo 1 || echo 0) \
    -E ENN_PBUF_FAULTS=${FAULTS:-0} -E ENN_ADD_TIMERS=${TIMERS:-0} -E ENN_LOG_GROW_LINEAR=${LOG_LINEAR:-0} -E ENN_LOG_PACE=${LOG_PACE:-1} -E WINDOW=${WINDOW:-1e5} -E PYTHONPATH=$(realpath ${PP:-src}) python $H $D $MAXN $W $F ${DEV:-nvme0n1} > ${F%.jsonl}.log 2>&1
  rm -rf $W; echo DONE $a $(date +%T) >> $O/sweep.log
done
