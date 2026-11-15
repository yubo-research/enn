#!/bin/bash
# Requires reports/report-0240-data/patches/node_buf.patch (git apply); the ENN_NODE_BUF flag is not in HEAD.
# Usage: nodebuf_ram_ab.sh OUT D N REPS PYTHONPATH ARM...   (report 0240; 1G cap, 1 CPU, store on /mnt/scratch)
# Paired in-RAM add/query timing (advise_data_ab.py) per ENN_NODE_BUF arm (suffix L = leaf parents only), arms alternating per seed.
O=$1; D=$2; N=$3; R=$4; P=$5; shift 5; H=$(cd "$(dirname "$0")" && pwd)/advise_data_ab.py
for s in $(seq 1 $R); do for a in "$@"; do
  W=/mnt/scratch/tmp/nb_ram; rm -rf $W
  timeout 900 systemd-run --user --scope -q -p MemoryMax=1G -p MemoryHigh=1G -p CPUQuota=100% \
    -E ENN_NODE_BUF=${a%L} -E ENN_NODE_BUF_LEAF=$([ "${a%L}" != "$a" ] && echo 1 || echo 0) -E SEED=$s -E PYTHONPATH=$P python $H $D $N $W $O
  rm -rf $W
done; done
