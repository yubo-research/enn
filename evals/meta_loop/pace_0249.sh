#!/bin/bash
# Usage: pace_0249.sh OUTDIR CAP MAXN REPS ARM...   (repo root; report 0249)
# Capped add sweeps (stall_probe.sh, D=100, EBS, 1e6-row windows) of log pacing off/on, arms alternated per rep.
# Arm = <pbuf arm>_<LOG_PACE>, e.g. 3000aph_0 3000aph_1. dirty_files.py samples per-file dirty pages at 1 Hz beside each run.
# PP = PYTHONPATH overlay (one build for both arms). Output rep<i>/stall_D100_<CAP>_pb<arm>.jsonl, dirty_<arm>.jsonl.
O=$1; C=$2; M=$3; R=$4; shift 4; H=$(cd "$(dirname "$0")" && pwd)
for r in $(seq 1 $R); do for a in "$@"; do
  pb=${a%_*}; lp=${a##*_}; mkdir -p $O/rep$r
  python $H/dirty_files.py /home/dsweet/bench_work/stall_$pb $O/rep$r/dirty_$a.jsonl 1 > /dev/null 2>&1 3>&- 4>&- &
  S=$!
  LOG_PACE=$lp WINDOW=1e6 SECS=${SECS:-5400} bash $H/stall_probe.sh $O/rep$r.tmp 100 $M $C $pb
  kill $S 2>/dev/null; wait $S 2>/dev/null
  mv $O/rep$r.tmp/stall_D100_${C}_pb$pb.jsonl $O/rep$r/stall_D100_${C}_pb$a.jsonl
  mv $O/rep$r.tmp/stall_D100_${C}_pb$pb.log $O/rep$r/stall_D100_${C}_pb$a.log; rm -rf $O/rep$r.tmp
done; done
