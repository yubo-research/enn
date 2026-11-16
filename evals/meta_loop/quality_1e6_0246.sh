#!/bin/bash
# Usage: quality_1e6_0246.sh OUTDIR D N SEEDS...   (repo root; report 0246)
# Paired quality/timing at one (D, N) for three arms under the 1 GB scope: pbuf (every-level buffers, 1000aph),
# head0 (HEAD defaults, no buffers), main (/mnt/scratch/enn_bench/main). Appends to OUTDIR/{pbuf,head0,main}.jsonl.
O=$(realpath $1); D=$2; N=$3; shift 3; R=$(realpath evals/meta_loop/quality_cmp.py); M=${MAIN:-/mnt/scratch/enn_bench/main}
run() { W=/home/dsweet/bench_work/q46_$1_$S; rm -rf $W; (cd $2 && env $3 PYTHONPATH=src systemd-run --user --scope -q -p MemoryMax=1G -p MemoryHigh=1G -p CPUQuota=100% python $R $D $N $S $W $O/$1.jsonl >> $O/$1.log 2>&1); rm -rf $W; }
for S in "$@"; do
  run pbuf . "ENN_PBUF=1000 ENN_PBUF_ALL=1 ENN_PBUF_PREFETCH=1 ENN_PBUF_PUNCH=1"
  run head0 . "ENN_PBUF=0"
  run main $M "ENN_PBUF=0"
done
echo DONE >> $O/q46.log
