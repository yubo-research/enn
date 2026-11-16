#!/bin/bash
# Usage: extent_cap.sh OUTDIR REPS [ARMS]   (repo root; report 0244)
# Needs reports/report-0244-data/patches/leaf_extent.patch for the x arm (reverted in 0244).
# 256M-capped add sweep to 1e7 rows, D=100, EBS, via pbuf_cap.sh; arms (default "1000ax 1000a")
# interleaved per rep; rep r goes to OUTDIR/r<r>/.
O=$1; R=$2; ARMS=${3:-"1000ax 1000a"}; H=$(cd "$(dirname "$0")" && pwd)
for r in $(seq 1 $R); do for a in $ARMS; do
  bash $H/pbuf_cap.sh $O/r$r 100 1e7 256M 3600 $a
done; done
