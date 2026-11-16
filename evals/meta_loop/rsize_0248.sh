#!/bin/bash
# Usage: rsize_0248.sh OUTDIR CAP MAXN REPS ARM...   (repo root; report 0248)
# Capped add sweeps (stall_probe.sh, D=100, EBS) with 1e6-row windows, arms alternated per rep: rep<i>/stall_D100_<CAP>_pb<arm>.jsonl
O=$1; C=$2; M=$3; R=$4; shift 4
for r in $(seq 1 $R); do for a in "$@"; do
  WINDOW=1e6 SECS=${SECS:-5400} bash "$(dirname "$0")"/stall_probe.sh $O/rep$r 100 $M $C $a
done; done
