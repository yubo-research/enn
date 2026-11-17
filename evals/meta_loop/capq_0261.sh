#!/bin/bash
# Usage: capq_0261.sh OUTDIR D MAXN CAP SECONDS "name:lloyd_every:lloyd_graph:walk_graph:leaf_coef:pbuf:lloyd_flush ..."   (report 0261)
# As capq_0260.sh plus buffers: pbuf 0 = unbuffered; pbuf R>0 = every-level buffers of R rows (ENN_PBUF=R ENN_PBUF_ALL=1,
# prefetch, hole punch, paced log; the 0248-0250 configuration). lloyd_flush 1 = ENN_LLOYD_FLUSH=1 (Lloyd passes at buffer flush).
# XV=VAR=VAL passes one extra env var. OV = overlay dir under /home/dsweet/bench_work (default ov_0261). Arms run one after another, never concurrently.
O=$1; D=$2; MAXN=$3; C=$4; S=$5; ARMS=$6; mkdir -p $O; O=$(realpath $O)
H=$(cd "$(dirname "$0")" && pwd)/capq_0260.py
for a in $ARMS; do
  IFS=: read name lloyd graph walk leaf pbuf flush <<< "$a"
  on=$([[ $pbuf == 0 ]] && echo 0 || echo 1)
  W=/home/dsweet/bench_work/capq0261_$name; rm -rf $W; F=$O/capq_D${D}_${C}_$name${REP}.jsonl; rm -f $F
  timeout $S systemd-run --user --scope -q -p MemoryMax=$C -p MemoryHigh=$C -p CPUQuota=100% \
    -E ENN_PBUF=$pbuf -E ENN_LOG_PACE=1 -E ENN_PBUF_ALL=$on -E ENN_PBUF_PREFETCH=$on -E ENN_PBUF_PUNCH=$on -E FINCORE=1 \
    -E ENN_LLOYD_EVERY=$lloyd -E ENN_LLOYD_GRAPH=$graph -E ENN_WALK_GRAPH=$walk -E LEAF_COEF=$leaf -E ENN_LLOYD_FLUSH=$flush ${XV:+-E $XV} \
    -E NQ=${NQ:-$([[ $C == 1G ]] && echo 30 || echo 100)} -E NF=${NF:-$([[ $C == 1G ]] && echo 5 || echo 10)} \
    -E PYTHONPATH=/home/dsweet/bench_work/${OV:-ov_0261} \
    python $H $D $MAXN $W $F nvme0n1p1 > ${F%.jsonl}.log 2>&1
  rm -rf $W
  echo DONE $name$REP $(date +%T) >> $O/sweep.log
done
