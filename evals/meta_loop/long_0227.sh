#!/bin/bash
# Usage: long_0227.sh OUTDIR CAP MAXN TIMEOUT_S DISKLV -- hybrid buffers: levels > DISKLV in RAM, levels <= DISKLV on disk.
O=$1; C=$2; MAXN=$3; T=$4; export DISKLV=$5; H=$(cd "$(dirname "$0")" && pwd)
d=$O/hyb_d$DISKLV; mkdir -p $d
bash $H/buffer_sim.sh $d $C $MAXN $T buffered 4096 16 hyb
echo ok > $d/done.txt
