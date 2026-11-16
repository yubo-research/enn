cd /home/dsweet/projects/enn
for spec in 256:_c256_r1 64:_c64_r1 64:_c64_r2 256:_c256_r2; do c=${spec%%:*}; r=${spec#*:}
  PP=/home/dsweet/bench_work/ov_c$c REP=$r bash evals/meta_loop/capq_0250.sh /home/dsweet/projects/enn/reports/report-0254-data/capq 10 3e6 256M 3600 0
done
echo CHAIN_DONE >> /home/dsweet/projects/enn/reports/report-0254-data/capq/sweep.log
