#!/bin/bash
# Usage: long_0229.sh OUTDIR CAP MAXN TIMEOUT_S PAUSE_S -- writeback test: fsync+sleep, then re-time buf vs leaf reads (hybrid d2).
O=$1; C=$2; MAXN=$3; T=$4; export PAUSE=$5; export DISKLV=2; export FS=64; H=$(cd "$(dirname "$0")" && pwd)
d=$O/pause; mkdir -p $d
bash $H/buffer_sim.sh $d $C $MAXN $T buffered 4096 16 hyb
echo ok > $d/done.txt
