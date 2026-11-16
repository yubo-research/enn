#!/bin/bash
# Usage: pread_ab_0250.sh OUTDIR PP REPS N...   (repo root; report 0250; PP must be a build with reports/report-0250-data/patches/row_pread.patch, reverted)
# Capped (1G, 1 CPU, EBS) query attribution (qattr_0250.py, 100 posterior queries x 2 passes) with
# ENN_ROW_PREAD=0 vs 1 on one build (PP), no add buffers. Arms alternate within each rep.
O=$1; P=$2; R=$3; shift 3; mkdir -p $O
for r in $(seq 1 $R); do for N in "$@"; do
  for a in $([ $((r % 2)) = 1 ] && echo "0 1" || echo "1 0"); do
    W=/home/dsweet/bench_work/pread_$a; rm -rf $W
    timeout 3600 systemd-run --user --scope -q -p MemoryMax=1G -p MemoryHigh=1G -p CPUQuota=100% \
      -E ENN_ROW_PREAD=$a -E NQ=100 -E PYTHONPATH=$P:evals/meta_loop \
      python evals/meta_loop/qattr_0250.py 100 $N $W $O/pread${a}_${N}_r$r.json nvme0n1p1 base again > $O/pread${a}_${N}_r$r.log 2>&1
    rm -rf $W
  done
done; done
echo DONE >> $O/done.txt
