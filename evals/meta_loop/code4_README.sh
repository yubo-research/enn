#!/bin/bash
# Report 0237 harness index (4-bit page codes, ENN_CODE4). Run from repo root.
#   code4_ab.sh OUT D MAXN CAP SECONDS REPS      cold+warm query per N, one process per arm (code4_ab.py ARM D MAXN WORK OUT DEV).
#                                                 env PP = PYTHONPATH of build; INTRINSIC=m for data near an m-dim subspace.
#   code4_stats.py DIR D CAP REP...               paired per-N summary of code4_ab output (Wilcoxon on query index).
#   code4_ram_ab.sh OUT D N REPS BASE_PP HEAD_PP  in-RAM add/query, arms base / off / on (advise_data_ab.py).
#   code4_ram_stats.py FILE                       paired t-tests vs base.
#   code4_base_cold.sh OUT D MAXN CAP BASE HEAD   cold A/B of two builds, default path (ENN_CODE4=0).
#   code4_quality.sh OUT D "NS" SEEDS PP          recall/NRMSE/loglik both arms (recall_cmp.py; env INTRINSIC, CODE_EXTRA).
#   fig_0237.py                                   figure for report 0237.
echo "see comments in $0"
