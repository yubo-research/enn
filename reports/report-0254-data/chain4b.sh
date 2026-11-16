cd /home/dsweet/projects/enn
O=/home/dsweet/projects/enn/reports/report-0254-data/qual
H=evals/meta_loop
for s in 0 1 2 3 4; do
  for arm in l30 main; do W=/mnt/scratch/lq_c4; rm -rf $W
    if [ $arm = main ]; then PP=/mnt/scratch/enn_bench/main/src; L=0; N=1e5; OUT=$O/main_1e5/main.jsonl; else PP=/home/dsweet/bench_work/ov_c256; L=30; N=1e6; OUT=$O/d10_1e6/c256_l30.jsonl; fi
    mkdir -p $(dirname $OUT)
    timeout 3000 systemd-run --user --scope -q -p MemoryMax=2G -p MemoryHigh=2G -p CPUQuota=100% -E PYTHONPATH=$PP python $H/recall_cmp.py 10 $N $s $L $W $OUT > /dev/null 2>> $O/err4.log
    rm -rf $W
  done
done
echo CHAIN4_DONE >> $O/err4.log
