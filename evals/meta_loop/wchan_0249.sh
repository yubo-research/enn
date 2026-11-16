#!/bin/bash
# Usage: wchan_0249.sh OUTDIR CAP MAXN ARM...   (repo root; report 0249)
# Wait-channel sampling (wchan_sample.py, 200 Hz) beside a capped stall_probe.sh run (D=100, EBS, 1e6-row windows).
# Arm = <pbuf arm>_<LOG_PACE>, e.g. 3000aph_0 3000aph_1. Output: wchan_<arm>.json, stall_D100_<CAP>_pb<arm>.jsonl.
O=$1; C=$2; M=$3; shift 3; H=$(cd "$(dirname "$0")" && pwd); mkdir -p $O
for a in "$@"; do
  pb=${a%_*}; lp=${a##*_}; T=$O/tmp_$a; rm -rf $T; mkdir -p $T
  python $H/wchan_sample.py stall_probe.py $T/stall_D100_${C}_pb$pb.jsonl $O/wchan_$a.json > $O/wchan_$a.log 2>&1 3>&- 4>&- &
  S=$!
  LOG_PACE=$lp WINDOW=1e6 SECS=${SECS:-5400} bash $H/stall_probe.sh $T 100 $M $C $pb
  wait $S
  mv $T/stall_D100_${C}_pb$pb.jsonl $O/stall_D100_${C}_pb$a.jsonl; mv $T/stall_D100_${C}_pb$pb.log $O/stall_D100_${C}_pb$a.log; rm -rf $T
done
