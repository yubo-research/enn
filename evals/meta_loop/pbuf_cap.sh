#!/bin/bash
# Usage: pbuf_cap.sh OUTDIR D MAXN CAP SECONDS ARM...   (run from repo root; report 0242)
# Arm suffixes (report 0243): a = ENN_PBUF_ALL=1 (buffers at every level), s = ENN_PBUF_STAGGER=1; e.g. 1000as.
# Capped add sweep (bytes_per_add.py, FINCORE residency every ~30 s) per leaf-parent buffer size,
# sequential, 1 CPU, store on the root disk. Arm = ENN_PBUF rows (0 = no buffers). PP = PYTHONPATH (default src).
O=$1; D=$2; MAXN=$3; C=$4; S=$5; shift 5; mkdir -p $O; O=$(realpath $O)
H=$(cd "$(dirname "$0")" && pwd)/bytes_per_add.py
for a in "$@"; do
  W=/home/dsweet/bench_work/pbcap_$a; rm -rf $W; F=$O/bpa_D${D}_${C}_pb$a.jsonl; rm -f $F
  timeout $S systemd-run --user --scope -q -p MemoryMax=$C -p MemoryHigh=$C -p CPUQuota=100% \
    -E ENN_PBUF=${a%%[as]*} -E ENN_PBUF_ALL=$([[ $a == *a* ]] && echo 1 || echo 0) -E ENN_PBUF_STAGGER=$([[ $a == *s* ]] && echo 1 || echo 0) -E FINCORE=1 -E PYTHONPATH=${PP:-src} \
    python $H $D $MAXN $W $F nvme0n1p1 > ${F%.jsonl}.log 2>&1
  rm -rf $W
  echo DONE $a $(date +%T) >> $O/sweep.log
done
