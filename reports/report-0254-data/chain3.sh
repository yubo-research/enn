cd /home/dsweet/projects/enn
O=/home/dsweet/projects/enn/reports/report-0254-data/capq100
mkdir -p $O
until grep -q CHAIN2_DONE /home/dsweet/projects/enn/reports/report-0254-data/capq/sweep.log; do sleep 10; done
for spec in 256:_c256_r1 64:_c64_r1; do c=${spec%%:*}; r=${spec#*:}
  PP=/home/dsweet/bench_work/ov_c$c REP=$r bash evals/meta_loop/capq_0250.sh $O 100 3e6 256M 3600 0
done
echo CHAIN3_DONE >> $O/sweep.log
