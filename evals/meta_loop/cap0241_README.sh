#!/bin/bash
# Report 0241 harness index (add-cliff onset vs code format; cold read cost vs size). Run from repo root.
#   code4_cap.sh OUT D MAXN CAP SECONDS ARM...   capped add sweep (bytes_per_add.py, FINCORE residency per file every ~30 s),
#                                                 one arm at a time, 1 CPU, store on root disk. ARM = ENN_CODE4 value
#                                                 (0 8-bit, 1 raw 4-bit, 2 residual 4-bit); suffix i = ENN_RESID_INHERIT=1, tX = ENN_RESID_TAU=X.
#   disk_extent.py DIR [FILE_MIB] [READS]         O_DIRECT QD1 pread of S KiB at random S-aligned offsets of a fresh file
#                                                 (true device reads). env SIZES_KIB (default 4..1024), REPS (default 2).
#   fig_0241.py DATA_DIR OUT.pdf                  figure + onset table. DATA_DIR holds cap256/bpa_D100_256M_c{0,2i,2}.jsonl and
#                                                 extent/{ebs,ebs2,nvme}.jsonl. Onset = first sample >= 100 us/row. Hot bytes/row =
#                                                 resident codes+slots+meta+train_y / N at the first fincore sample (pre-eviction);
#                                                 predicted onset = 200 MiB / hot bytes (not applied to arm 2: train_x re-reads fill the cache).
# Example (report 0241):
#   evals/meta_loop/code4_cap.sh reports/report-0241-data/cap256 100 1e7 256M 2400 0 2i 2
#   python evals/meta_loop/disk_extent.py /home/dsweet/bench_work/ext > ebs.jsonl
#   python evals/meta_loop/fig_0241.py reports/report-0241-data fig_0241.pdf
