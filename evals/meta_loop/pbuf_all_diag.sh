#!/bin/bash
# Usage: pbuf_all_diag.sh OUTDIR D MAXN CAP DISKDIR DEV ARM...   (repo root; report 0243)
# pbuf_diag.py per arm under a cgroup cap, 1 CPU. Arm = ENN_PBUF rows, then optional suffixes:
# a = ENN_PBUF_ALL=1 (buffers at every level), s = ENN_PBUF_STAGGER=1. E.g. 1000, 1000a, 1000as.
# DISKDIR/DEV: /mnt/scratch nvme1n1 (local NVMe) or /home/dsweet/bench_work nvme0n1p1 (EBS). PP = PYTHONPATH (default src).
O=$1; D=$2; MAXN=$3; C=$4; DD=$5; DEV=$6; shift 6; mkdir -p $O; O=$(realpath $O)
H=$(cd "$(dirname "$0")" && pwd)/pbuf_diag.py; P=$(realpath ${PP:-src})
for a in "$@"; do
  r=${a%%[as]*}; all=0; st=0; [[ $a == *a* ]] && all=1; [[ $a == *s* ]] && st=1
  W=$DD/pbadiag_$a; rm -rf $W; F=$O/diag_D${D}_${C}_pb$a.jsonl; rm -f $F
  (cd /tmp && timeout ${SECS:-7200} systemd-run --user --scope -q -p MemoryMax=$C -p MemoryHigh=$C -p CPUQuota=100% \
    -E ENN_PBUF=$r -E ENN_PBUF_ALL=$all -E ENN_PBUF_STAGGER=$st -E WINDOW=${WINDOW:-50000} -E PYTHONPATH=$P:$(dirname $H) \
    python $H $D $MAXN $W $F $DEV > ${F%.jsonl}.log 2>&1)
  rm -rf $W
  echo DONE $a $(date +%T) >> $O/sweep.log
done
