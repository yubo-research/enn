#!/bin/bash
# Usage: pbuf_ram_ab.sh OUT D N REPS   (repo root; 1G cap, 1 CPU; report 0242)
# Arm suffixes (report 0243): a = ENN_PBUF_ALL=1 (buffers at every level), s = ENN_PBUF_STAGGER=1; e.g. 1000as. x = ENN_LEAF_EXTENT=1 (needs reports/report-0244-data/patches/leaf_extent.patch), p = ENN_PBUF_PREFETCH=1 (0244; without p the arm sets 0), h = ENN_PBUF_PUNCH=1 (0245; hole-punch taken buffer pages).
# Paired in-RAM add/query timing (advise_data_ab.py, uniform data) of ENN_PBUF arms in $ARMS (default "0 1000")
# RDIR = work dir parent (default /home/dsweet/bench_work = EBS); on one build (PP = PYTHONPATH, default src), order rotated per seed; arm label "pb_<arm>" in field "advise". Stats: pbuf_ram_stats.py OUT pb_0
O=$(realpath -m $1); D=$2; N=$3; R=$4; P=$(realpath ${PP:-src}); H=$(cd "$(dirname "$0")" && pwd)/advise_data_ab.py
read -ra A <<< "${ARMS:-0 1000}"
for s in $(seq 1 $R); do for j in $(seq 0 $((${#A[@]} - 1))); do
  a=${A[$(( (s + j) % ${#A[@]} ))]}
  W=${RDIR:-/home/dsweet/bench_work}/pbram; rm -rf $W
  (cd /tmp && timeout 900 systemd-run --user --scope -q -p MemoryMax=1G -p MemoryHigh=1G -p CPUQuota=100% \
    -E ENN_DATA_ADVISE=pb_$a -E ENN_PBUF=${a%%[asxph]*} -E ENN_PBUF_ALL=$([[ $a == *a* ]] && echo 1 || echo 0) -E ENN_PBUF_STAGGER=$([[ $a == *s* ]] && echo 1 || echo 0) -E ENN_LEAF_EXTENT=$([[ $a == *x* ]] && echo 1 || echo 0) -E ENN_PBUF_PREFETCH=$([[ $a == *p* ]] && echo 1 || echo 0) -E ENN_PBUF_PUNCH=$([[ $a == *h* ]] && echo 1 || echo 0) -E SEED=$s -E PYTHONPATH=$P python $H $D $N $W $O)
  rm -rf $W
done; done
