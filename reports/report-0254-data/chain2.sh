cd /home/dsweet/projects/enn
O=/home/dsweet/projects/enn/reports/report-0254-data
until grep -q CHAIN_DONE $O/capq/sweep.log; do sleep 10; done
bash evals/meta_loop/leafcap_quality_0254.sh $O/qual/d10_1e5 10 1e5 10 "256 64"
INTRINSIC=5 bash evals/meta_loop/leafcap_quality_0254.sh $O/qual/d100i5_1e5 100 1e5 5 "256 64"
bash evals/meta_loop/leafcap_quality_0254.sh $O/qual/d10_1e6 10 1e6 5 "256"
echo QUAL_DONE >> $O/capq/sweep.log
PP=/home/dsweet/bench_work/ov_c256 REP=_c256_big bash evals/meta_loop/capq_0250.sh $O/capq 10 3e7 256M 7000 0
echo CHAIN2_DONE >> $O/capq/sweep.log
