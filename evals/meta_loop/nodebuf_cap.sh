#!/bin/bash
# Requires reports/report-0240-data/patches/node_buf.patch (git apply); the ENN_NODE_BUF flag is not in HEAD.
# Usage: nodebuf_cap.sh OUTDIR D MAXN CAP SECONDS PYTHONPATH ARM...   (report 0240)
# Capped add sweep (bytes_per_add.py) per ENN_NODE_BUF arm (0 = off; suffix L = ENN_NODE_BUF_LEAF=1,
# buffers at leaf parents only), sequential, 1 CPU, store on the root disk.
O=$1; D=$2; MAXN=$3; C=$4; S=$5; P=$6; shift 6; mkdir -p $O; O=$(realpath $O)
H=$(cd "$(dirname "$0")" && pwd)/bytes_per_add.py
for a in "$@"; do
  W=/home/dsweet/bench_work/nb_$a; rm -rf $W; F=$O/bpa_D${D}_${C}_b$a.jsonl; rm -f $F
  timeout $S systemd-run --user --scope -q -p MemoryMax=$C -p MemoryHigh=$C -p CPUQuota=100% \
    -E ENN_NODE_BUF=${a%L} -E ENN_NODE_BUF_LEAF=$([ "${a%L}" != "$a" ] && echo 1 || echo 0) -E PYTHONPATH=$P python $H $D $MAXN $W $F nvme0n1p1 > ${F%.jsonl}.log 2>&1
  rm -rf $W
  echo DONE $a >> $O/sweep.log
done
