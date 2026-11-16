#!/bin/bash
# Usage: capq_0250.sh OUTDIR D MAXN CAP SECONDS ARM...   (run from repo root; report 0250)
# Capped add + query sweep (capq_0250.py): per checkpoint, add time and NQ posterior + NF function-draw
# query times under the cgroup cap, 1 CPU, store on the root disk. Arms as in pbuf_cap.sh:
# 0 = no buffers (default build); 3000aph = ENN_PBUF=3000 + every level (a) + prefetch (p) + punch (h); u = unpaced log.
# NQ/NF default 100/10 (30/5 when CAP=1G). PP = PYTHONPATH (default src). REP = suffix for the output file name (default empty).
O=$1; D=$2; MAXN=$3; C=$4; S=$5; shift 5; mkdir -p $O; O=$(realpath $O)
H=$(cd "$(dirname "$0")" && pwd)/capq_0250.py
for a in "$@"; do
  W=/home/dsweet/bench_work/capq_$a; rm -rf $W; F=$O/capq_D${D}_${C}_pb$a${REP}.jsonl; rm -f $F
  timeout $S systemd-run --user --scope -q -p MemoryMax=$C -p MemoryHigh=$C -p CPUQuota=100% \
    -E ENN_PBUF=${a%%[asxphu]*} -E ENN_LOG_PACE=$([[ $a == *u* ]] && echo 0 || echo 1) -E ENN_PBUF_ALL=$([[ $a == *a* ]] && echo 1 || echo 0) -E ENN_PBUF_PREFETCH=$([[ $a == *p* ]] && echo 1 || echo 0) -E ENN_PBUF_PUNCH=$([[ $a == *h* ]] && echo 1 || echo 0) -E FINCORE=1 -E NQ=${NQ:-$([[ $C == 1G ]] && echo 30 || echo 100)} -E NF=${NF:-$([[ $C == 1G ]] && echo 5 || echo 10)} -E PYTHONPATH=${PP:-src} \
    python $H $D $MAXN $W $F nvme0n1p1 > ${F%.jsonl}.log 2>&1
  rm -rf $W
  echo DONE $a$REP $(date +%T) >> $O/sweep.log
done
