#!/bin/bash
# Usage: long_0228.sh OUTDIR CAP MAXN TIMEOUT_S DISKLV FS -- per-level query pread timing, hybrid buffers.
O=$1; C=$2; MAXN=$3; T=$4; export DISKLV=$5; export FS=$6; H=$(cd "$(dirname "$0")" && pwd)
d=$O/hyb_d${DISKLV}_fs$FS; mkdir -p $d
bash $H/buffer_sim.sh $d $C $MAXN $T buffered 4096 16 hyb
echo ok > $d/done.txt
