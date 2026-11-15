#!/bin/bash
# Usage: pbuf_diag.sh OUTDIR D MAXN CAP OVERLAY ARM...   (repo root; report 0242; store on local NVMe /mnt/scratch)
# pbuf_diag.py per arm under a cgroup cap, 1 CPU. Arm = ENN_PBUF rows, suffix n = ENN_PBUF_ADVISE=0, then r = ENN_PBUF_ROOT=1 (e.g. 1000nr).
O=$1; D=$2; MAXN=$3; C=$4; P=$5; shift 5; mkdir -p $O; O=$(realpath $O); H=$(cd "$(dirname "$0")" && pwd)/pbuf_diag.py
for a in "$@"; do
  b=${a%r}; root=0; [ "$b" != "$a" ] && root=1; r=${b%n}; adv=1; [ "$r" != "$b" ] && adv=0
  W=/mnt/scratch/pbdiag_$a; rm -rf $W; F=$O/diag_D${D}_${C}_pb$a.jsonl; rm -f $F
  (cd /tmp && timeout 3600 systemd-run --user --scope -q -p MemoryMax=$C -p MemoryHigh=$C -p CPUQuota=100% \
    -E ENN_PBUF=$r -E ENN_PBUF_ADVISE=$adv -E ENN_PBUF_ROOT=$root -E PYTHONPATH=$P:$(dirname $H) python $H $D $MAXN $W $F nvme1n1 > ${F%.jsonl}.log 2>&1)
  rm -rf $W
done
