#!/bin/bash
# Usage: long_0225.sh OUTDIR CAP MAXN TIMEOUT_S -- 0224 'mrg' run (4096-row buffers, F=16) that also
# records leaf_only_us (one 8 KB leaf pread, same random leaf distribution) and preads_per_query.
O=$1; C=$2; MAXN=$3; T=$4; H=$(cd "$(dirname "$0")" && pwd)
d=$O/long_c4096_f16; mkdir -p $d
bash $H/buffer_sim.sh $d $C $MAXN $T buffered 4096 16 mrg
echo ok > $O/long_done.txt
