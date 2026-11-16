#!/bin/bash
# report 0250 run chain (from repo root)
S=evals/meta_loop/capq_0250.sh
$S reports/report-0250-data/cap256 100 1e7 256M 3600 3000aph
$S reports/report-0250-data/cap256 100 3e6 256M 3600 0
$S reports/report-0250-data/cap1G 100 3e7 1G 5400 3000aph
$S reports/report-0250-data/cap1G 100 3e6 1G 3600 0
echo CHAIN_DONE $(date +%T) >> reports/report-0250-data/chain.log
