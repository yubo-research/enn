#!/bin/bash
# Report 0238 harness index (residual 4-bit page codes ENN_CODE4=2; refine-more certification ENN_REFINE_MORE). Repo root.
#   resid_ab.sh OUT D MAXN CAP SECONDS REPS        cold+warm query per N, one process per arm (code4_ab.py); env ARMS
#                                                   (e.g. "0 1 2 2n"; suffix n = ENN_REFINE_MORE=0), PP, INTRINSIC, LD_LIBRARY_PATH.
#   resid_stats.py DIR D CAP BASE_ARM ARM...       paired per-N cold summary (Wilcoxon on log ratio per query).
#   resid_retry_diag.py ARM D MAXN WORK OUT        warm f32 re-walks and exact rows per query (env ENN_CODE4, INTRINSIC).
#   resid_quality.sh OUT D "NS" SEEDS PP           recall/NRMSE/loglik per arm (recall_cmp.py); paired_quality.py to compare.
#   resid_ram_ab.sh OUT D N REPS PP                in-RAM add/query per arm on one build (env ARMS); resid_ram_stats.py OUT.
#   resid_base_ram.sh OUT D N REPS BASE HEAD       default-path in-RAM check: base build vs head (refine-more on/off).
#   resid_ram_stats.py FILE                        paired t-tests vs c4_0 (or c4_base).
#   resid_slack_sim.py D INTRINSIC N...            offline model: code-error slack / leaf radius for 8-bit, raw 4-bit, residual.
#   fig_0238.py DATA_DIR OUT.pdf                   figure for report 0238 (reports/report-0238-data).
# Builds: wheel overlays (maturin build --auditwheel skip; src/enn + wheel .so + ennbo.libs) under /mnt/scratch/tmp.
echo "see comments in $0"
