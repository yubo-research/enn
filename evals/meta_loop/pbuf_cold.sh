#!/bin/bash
# Usage: pbuf_cold.sh OUTDIR D MAXN CAP SECONDS ARM...   (run from repo root; report 0242)
# Arm suffixes (report 0243): a = ENN_PBUF_ALL=1 (buffers at every level), s = ENN_PBUF_STAGGER=1; e.g. 1000as. x = ENN_LEAF_EXTENT=1 (needs reports/report-0244-data/patches/leaf_extent.patch), p = ENN_PBUF_PREFETCH=1 (0244; without p the arm sets 0), h = ENN_PBUF_PUNCH=1 (0245/0246; without h the arm sets 0).
# True on-disk cost per add/query (cold_cost.py: store paged out before every op) per ENN_PBUF arm,
# sequential, cgroup cap, 1 CPU, store on the root disk. env INTRINSIC is not used by cold_cost.py (uniform sine data).
O=$1; D=$2; MAXN=$3; C=$4; S=$5; shift 5; mkdir -p $O; O=$(realpath $O)
H=$(cd "$(dirname "$0")" && pwd)/cold_cost.py
for a in "$@"; do
  W=/home/dsweet/bench_work/pbcold_$a; rm -rf $W; F=$O/cold_D${D}_${C}_pb$a.jsonl; rm -f $F
  timeout $S systemd-run --user --scope -q -p MemoryMax=$C -p MemoryHigh=$C -p CPUQuota=100% \
    -E ENN_PBUF=${a%%[asxph]*} -E ENN_PBUF_ALL=$([[ $a == *a* ]] && echo 1 || echo 0) -E ENN_PBUF_STAGGER=$([[ $a == *s* ]] && echo 1 || echo 0) -E ENN_LEAF_EXTENT=$([[ $a == *x* ]] && echo 1 || echo 0) -E ENN_PBUF_PREFETCH=$([[ $a == *p* ]] && echo 1 || echo 0) -E ENN_PBUF_PUNCH=$([[ $a == *h* ]] && echo 1 || echo 0) -E PYTHONPATH=${PP:-src} python $H $D $MAXN $W $F nvme0n1p1 > ${F%.jsonl}.log 2>&1
  rm -rf $W
  echo DONE cold $a $(date +%T) >> $O/sweep.log
done
