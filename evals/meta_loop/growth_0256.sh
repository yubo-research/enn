#!/bin/bash
O=$1; shift; mkdir -p $O; O=$(realpath $O)
H=$(cd "$(dirname "$0")" && pwd)/growth_0256.py
B=/home/dsweet/bench_work/g0256
D=${D:-10}; NQ=${NQ:-300}; I=${INTRINSIC:-0}
run() {
  local arm=$1 n=$2 seed=$3; local W=$B/w_${arm}_${n}_${seed}; rm -rf $W
  local pp=/home/dsweet/bench_work/ov_c64 cap=64 off="" ll=0 re=0
  case $arm in
    r64) off="64,256,1024"; [[ $(python -c "print(int(float('$n'))<=1000000)") == True ]] && ll=3 ;;
    r64r) re=16 ;;
    r256) pp=/home/dsweet/bench_work/ov_c256; cap=256 ;;
  esac
  PYTHONPATH=$pp ARM=$arm CAP=$cap OFFLINE=$off LLOYD=$ll ENN_REASSIGN=$re NQ=$NQ INTRINSIC=$I \
    python $H $D $n $seed $W $O/growth_D${D}_I${I}_${arm}.jsonl >> $O/growth_D${D}_I${I}_${arm}_${n}_${seed}.log 2>&1
  rm -rf $W
  echo "DONE $arm $n $seed $(date +%T)" >> $O/sweep.log
}
for spec in "$@"; do IFS=: read arm n seed <<< "$spec"; run $arm $n $seed; done
echo ALLDONE $(date +%T) >> $O/sweep.log
