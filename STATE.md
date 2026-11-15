# STATE (edit in place; reports are the log)

## Established (1 GB cgroup cap unless noted; "branch" = dsweet/agg_obs, HEAD b2d87fa + uncommitted 0237 changes; "main" = 3246c4e)
- Queries: branch faster than main at N>=1e5 (D=10: 0.22 vs 0.89 s; D=100: 2.2 vs 3.2 s); main faster at N<=1e3. (0216)
- Memory: branch RssAnon 21 MB vs main 558 MB at D=100, N=1e6. (0216)
- Quality vs main, 10 paired seeds (0217): identical at N<=1e4. At N=1e5 branch is significantly worse: loglik -0.022 (SE 0.005, p=0.002) at D=10; -0.008 (SE 0.003, p=0.017) at D=100. NRMSE +0.002 (D=10), +0.008 (D=100). (0216's "not significant" was 3 seeds.)
- Cause at D=10: tree recall@10 = 0.958 at leaf coefficient c=10. Exact-kNN NRMSE 0.465 vs approx 0.467. With c=30 recall 0.997 and loglik gap closes (+0.001, SE 0.003, p=0.76; 5 seeds). (0217)
- At D=100 recall is only 0.18 (c=10), but exact kNN gives NRMSE ~1.01: this test function has no learnable structure there, so quality at D=100 cannot judge the search. (0217)
- In-RAM add: main 2.5-3x faster (gate build 2.0 vs 0.8 us/row, D=10, N=1e5). Past the cliff branch wins: 47 vs 766 us at D=100, N=3e6. Branch is NOT yet O(ln N): 12.5 us (1e6) -> 47 us (3e6) once store exceeds cap. (0216)
- Add time is dominated by disk waits beyond cache. (0162, 0168, 0173, 0177, 0215)
- Add-time cliff, D=100, caps 64M/256M/1G (single runs, 0218): 5-13 us/row plateau, then a 10-100x step at N ~ 1.5e5 / 1.1e6 / 2.2e6 (2-4k rows per MB of cap; store is 5-12x the cap at the step). Writes rise first (1.4 -> 5 KB/row at 1G), reads follow (0-11 KB/row). Hot set (0219, fincore, D=100, 256M): the cache fills with tree_codes.bin (165 MiB at N=1.02e6; meta 11, slots 23). Touched code pages grow ~176 B/row (1.4-1.5 pages per 8 KB leaf region), so N*~(C-60MiB)/176B~1.1e6 matches the 256M step. At 1G the formula gives 5.5e6 but the onset was 2.2e6 (untested: dirty-page limit). A fixed-size buffer only moves the cliff (flush gives each leaf 45M/N rows); per-level buffers are needed.
- Leaf coefficient L=30, 10 paired seeds, N<=1e5 (0218): D=10 quality parity with main at every N (dloglik -0.003, SE 0.002 at 1e5, p=0.32); query 0.50x main at 1e5 but 1.8x slower at 1e4 and 2.9x at 1e3. D=100 gap remains (-0.010, p=0.007; no skill there).
- Storage model (0220-0230; 256M cap, D=100 rows, single runs, no tree, no recall): in-place appends step 14.8 -> 349 us/row at N~2e6. Buffering removes the add step (4096-row buffers: 2-13 us/row to 1e7, slope ~2-6 us/lnN). Queries still rise 20-25x across the cap for every buffer design; one 8 KB cold leaf read goes 1.7 -> ~200 us there, so a cold-read step is paid by any disk design. Best design: one sorted run per buffer, 4.7-5.6x one leaf read; RAM upper levels (hybrid d2) cut adds 1.7x, not queries (~2.5 cold reads/query). Summary in 0231 (fig_0231.py); details 0220-0230.
- Buffer-file reads cost 1.2-1.35x leaf reads in the model. Not writeback (0229), not fragmentation (0230), not alignment (0229 data: 8K-aligned 396-402 vs leaf 298-326 us). Cold reads are equal across files (~670 us); gap is shared-cache hit rate (0230, rough).
- Five-metric scorecard, branch 84fba2d vs main 3246c4e, 1 GB cap, paired seeds (0226): branch does NOT beat main on all metrics. Add 2.0-6.1x slower at every N (D=10, N=1e6: 7.3 vs 1.2 us/row). Query 1.3-2.4x slower at N<=1e4, 0.39-0.69x at N>=1e5 (D=100). RSS 0.47x at D=100 N=1e6. Quality identical N<=1e4; D=100 N=1e5 dloglik -0.0105, p=0.007; N=1e6 underpowered (5 seeds).

- In-RAM add gap vs main (0232; 1 GB scope, 5 seeds, uncommitted AdviseGate on 61ed15c): the add-time prefetch (second routing descent + madvise per 32 adds) cost 0.5-3 us/row in RAM; morph is 0.03 us/row. Prefetch is useful past the cap (D=100, N=1.5e6, 256M, 1 run each: always 193, never 257, gated 187 us/row). Gating on major faults (>=8 per 32 adds, 32 groups hysteresis; ENN_ADVISE=off|always|gated) removes 65-83% of the overhead at N=1e5. Branch add is still 1.9-3.1x main (D=10 N=1e5: 1.99 vs 1.05; D=100: 7.51 vs 3.18); the rest is insert_row, unprofiled. Inside the malvin sandbox uncapped timings are ~3x slower than under systemd-run (hypothesis: syscall cost); use scope runs only.

- True on-disk cost (0235; store evicted before every op via MADV_PAGEOUT+fadvise, model kept open, 1G scope, 1 run, 20 queries/10 add groups per N): cold query 118/202/265 ms (D=10) and 246/355/413 ms (D=100) at N=1e4/1e5/1e6; cold add 1.1-2.7 ms/row (32-row groups, 9-40 KB read/row). Smooth from N=1, no step, consistent with a+b lnN for N>=1e4 (query slope 23+-11 / 28+-11 ms per lnN). In RAM it is 240-400x lower. A cold query reads ~2 pages per visited leaf (tree_codes + tree_meta; ~1250 pages at D=100 N=1e6), serially at ~0.33 ms/page. RssAnon flat 19-28 MB.
- Add cliff replicated (0234 data in 0235, D=100, 256M, 3 runs): fit N<=1e6 predicts 24 us/row at 3e6; measured 644+-5. The cliff is the jump from the RAM curve to the cold curve.
- Prefetch depth (0235, D=10, N=1e5, cold): default 202 ms, off 290, prefetch_pages=30 157, meta_prefetch=16 159, code_prefetch=30 291 (unexplained). Warm cost of deeper prefetch unmeasured.
- Where in-RAM add time goes (0233, timer build, 1 run, sandbox, N=1e5): insert_row 2.6 us/row (D=10), 5.7 (D=100). Splits 34%/45% (writes to old/new leaf dominate, alloc_near 0.23 us/row, median_split 0.1-0.2), descent 22%/16%, leaf push 10%/23%, morph 0.04. Shares only; timers inflate totals.

- Store disk = Amazon EBS (not local NVMe; 0235 mislabeled it), sustained ~3000 random 4K reads/s at any depth; QD1 486 us; local NVMe /mnt/scratch 99 us QD1, 68k/s at QD16 (0236, disk_qd.py). 0235's 0.33 ms/page is the EBS IOPS cap: cold query cost ~ pages x 0.33 ms; parallel reads can give <=1.46x here.
- Cold query time by file at N=1e6 (0236, keep_warm.sh; file kept resident): codes 67% (D=10) / 76% (D=100), meta 42% / 14%, x+y 0-8%.
- Leaf ids in code-page tail + child-is-leaf bit in parent route radii (0236; flag ENN_LEAF_IDS / set_leaf_ids, default on, only D<=50 = one-page code region): cold query -7..-17% at D=10 for 1e4<=N<=1e6 (sig at 4/5 sizes; 310->271 ms at 1e6), -11..-12% at D=30 N<=3e4 (n.s. above); pages -5..-13%; posteriors bit-identical. In RAM (N=1e5, 5 seeds): D=10 add 1.53->1.54 n.s., query 532->520 us (p=0.004); D=100 n.s. Uncommitted at 8b7809e, staged.

- 4-bit page codes (0237; flag ENN_CODE4=1, default OFF; tree_nib.rs; used only where they save code pages, D=100: 2->1 page): cold query 0.49-0.67x for 1e4<=N<=1e6 on uniform D=100 (465->273 ms at 1e6), but on D=100 data near a 5-dim subspace 1.2-1.7x SLOWER for 1e3<=N<=3e4, 0.63-0.68x only at N>=3e5; 1.05-1.1x at N<=3e3 (f32 re-walk retries). In RAM: query 2.1x (D=100). Quality parity after the bound fix (10 seeds; dloglik -0.004, p=0.23). Cause of mid-N loss (hypothesis, not proven): leaf slack = max row error ~10x the 8-bit bound, weak pruning (2.6x leaf rows at 1e4); not routing (exact f32 routing changed nothing).
- Default path after 0237 (8-bit, uncommitted): splits/morph move encoded code rows (set_leaf_coded) instead of decode+re-encode; leaf scan monomorphized per format. vs b2d87fa, 8 seeds N=1e5: D=100 add 0.85x (p=1e-4), query 0.96x (p=0.05); D=10 add 0.93x (p=0.09), query 0.99x n.s.; cold to 1e6 0.97-1.04, equal pages, identical posteriors.
- Leaves keep no f32 rows; codes are the only in-tree copy. Re-encoding decoded rows loses a stored error bound (0237); any future code format must move code bytes on split.
- Learnable D=100 test exists: INTRINSIC=5 in recall_cmp.py / code4_ab.py (x = z B + 0.01 noise, z in U[0,1]^5); exact-kNN NRMSE 0.16 at 1e5 vs 1.02 for the uniform test.
- Build hygiene: compare builds made the same way (maturin build wheels); develop vs wheel link different OpenBLAS (0237).

## Tried and rejected
- File fragmentation (0230), writeback contention (0229), 8K alignment (0229 data) as causes of slow buffer-file reads.
- Smaller fence granularity (16 vs 64 rows): only -10% query (0228).
- Fixed number of disk levels with RAM upper buffers: query unchanged, RAM grows with N (0227).
- Run-indexed buffers: queries 3.8x slower than full scan (0223). Per-child buffer segments: adds 5-10x slower (0221).
- Moving leaf metadata into the code page; one page per leaf (0219). Keeping the upper tree in memory (0173). Dropping the prefetch (0215).
- Leaf coefficient c=3: recall 0.79 at D=10 (0217).
- Wider AVX2 loop in l2_sq_f32 for D<32 (0233): no gain (1.87 vs 1.66 us/row insert).
- Morph work as the cause of the in-RAM add gap (0232).
- Leaf-id tail on the 2nd page of a 2-page code region (D=100): adds +8% in RAM (p=0.002), +25% cold at N<=30 (0236). Leaf test by depth (assumes balanced tree): 6635 vs 2708 pages at D=10 N=3e5, cause unproven; replaced by explicit parent bits (0236).
- 4-bit raw-coordinate codes as default (0237): mid-N cold loss on structured data, slower in RAM. Kept behind ENN_CODE4 (off).
- Tight certification boundary min(last, min_excluded(r_i - e_i)) (0237): no fewer retries; removed. Larger coded pool (CODE_EXTRA 100/300) does not raise 4-bit recall (0237).
- MADV_RANDOM on train_x/train_y mmaps (0235): cuts cold train_y reads 604 -> 40 KiB (k=10) but no cold time gain, and in-RAM adds +24% (D=100) / +52% (D=10), 5 paired seeds. Reverted; patch in reports/report-0235-data/patches/.

## Plan (priority: O(ln N) add and query with no cliff; quality never worse than main)
Per the 2026-11-15 ruling, judge every change first by cold cost (cold_cost.sh, leaf_ids_ab.sh), then by in-RAM cost (build_ab.sh) and the 0226 scorecard.
1. Lower the cold query curve. Done: 1b for D<=50 (0236); 1a tried as raw 4-bit codes (0237, off by default). Next: (a) residual 4-bit codes vs a per-leaf 8-bit anchor (IVF-PQ style) so leaf slack scales with leaf spread; judge with code4_ab.sh at INTRINSIC=5 and 0, D=100; (b) cheaper certification fallback than an f32 re-walk (refine more from train_x) so coarse codes do not lose at N<=3e3; (c) fewer leaves per query (recall vs leaves visited); (d) ids-first page for D>50. Parallel leaf reads only pay on NVMe.
2. Lower the cold add curve: buffered adds per level (B-epsilon / buffer tree, ideas.md P1).
3. Designed cache: pin top tree levels, plot cold cost vs pinned bytes; then repeat 0234-style capped runs (3 reps) to check the cliff is gone.
4. Open: in-RAM add gap vs main (2-6x, 0226) is the largest scorecard gap; deeper prefetch default; leaf coefficient 30 quality at 1e6; learnable D=100 test function (0217).
Harnesses: evals/meta_loop/ (code4_* see code4_README.sh, fig_0237.py, leaf_ids_ab.*, build_ab.sh, keep_warm.*, disk_qd.py, fig_0236.py, cold_cost.*, prefetch_cold.py, advise_data_ab.*, scorecard_0226.*, paired_quality.py, bytes_per_add.py, cap_sweep.sh, recall_cmp.py), scripts/bench_cmp/bench.py.

## Questions for the operator
- Which disk is the target? The store has been on EBS (3000 IOPS cap, 0236), where cold cost is IOPS-bound; the local NVMe (/mnt/scratch) is ~20x faster and rewards parallel reads. Should cold-cost judgments use one, the other, or both?
- May the code-page format change (ids before rows; smaller codes)? Old stores would need a rebuild or a version check (0236).
- Is "cold cost" as measured in 0235 (page-out of all store pages before each op, model kept open, heap state counts as cache) the yardstick you mean by "true on-disk cost"?
- Parallel reads (io_uring or WILLNEED batches) on the one calculation thread: acceptable under the one-thread rule? (matters only on NVMe, 0236)
- Must add-time parity with main hold in RAM, or only past the cliff? (0226) Rerun 1e6 quality with 10 seeds?
- Keep leaf coefficient 30 (closes D=10 gap, queries 1.8x slower than main at N=1e4)? (0218)
- Is a learnable-structure test function at D=100 acceptable for quality? (0217)
- Is a log-structured code store, or queries that skip buffers (relaxed read-your-writes), acceptable? (0219, 0223)
- AdviseGate (0232) was committed in a9e61af; thresholds (8 faults/32 adds, 32 groups) are untuned. Keep?
- Sandbox lacks perf (paranoid=4), gdb, valgrind; is a profiling-capable environment available? (0233)
- VISION-b.md does not exist; is another file intended?

- Commit the 0237 default-path changes (D=100 adds -15% in RAM, cold unchanged) together with the off-by-default ENN_CODE4 code, or commit only the default-path part? (0237)
- Should the D=100 quality yardstick switch to a learnable test (e.g., INTRINSIC=5 in recall_cmp.py)? The uniform one cannot detect neighbor-quality loss (0217, 0237).

## Operator ruling on design method (2026-11-15)
- The cliff at the cap is not a property of the problem. It comes from treating memory as the store: letting the page cache hold everything until it is full is not an algorithm, and shows no understanding of how the data is used. Stop treating "a step of cold-read size is paid by any disk design" (0224, 0225, 0231) as settled; that conclusion rests on a RAM-until-full design.
- Design order: (1) the best on-disk algorithm you can build, O(lnN) and fast, judged by its true on-disk cost from N=1 (below the cap, today's "disk reads" are really cache hits; 0225: 1.7 vs ~200 us). (2) Then use memory as a cache, choosing what to keep from measured access patterns: what every operation touches, and what the workload touches often. A cache chosen this way degrades gradually as N grows (contributing to the lnN slope), not all at once at the cap.
- Quality target is statistical parity with main (error bars, tests), not exactness; approximations are allowed.
- Beating main at small N is an engineering goal, not a conflict with the above.
