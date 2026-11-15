# STATE (edit in place; reports are the log)

## Established (1 GB cgroup cap unless noted; "branch" = dsweet/agg_obs, HEAD 4bdadb2; "main" = 3246c4e)
- Queries: branch faster than main at N>=1e5 (D=10: 0.22 vs 0.89 s; D=100: 2.2 vs 3.2 s); main faster at N<=1e3. (0216)
- Memory: branch RssAnon 21 MB vs main 558 MB at D=100, N=1e6. (0216)
- Quality vs main, 10 paired seeds (0217): identical at N<=1e4. At N=1e5 branch is significantly worse: loglik -0.022 (SE 0.005, p=0.002) at D=10; -0.008 (SE 0.003, p=0.017) at D=100. NRMSE +0.002 (D=10), +0.008 (D=100). (0216's "not significant" was 3 seeds.)
- Cause at D=10: tree recall@10 = 0.958 at leaf coefficient c=10. Exact-kNN NRMSE 0.465 vs approx 0.467. With c=30 recall 0.997 and loglik gap closes (+0.001, SE 0.003, p=0.76; 5 seeds). (0217)
- At D=100 recall is only 0.18 (c=10), but exact kNN gives NRMSE ~1.01: this test function has no learnable structure there, so quality at D=100 cannot judge the search. (0217)
- In-RAM add: main 2.5-3x faster (gate build 2.0 vs 0.8 us/row, D=10, N=1e5). Past the cliff branch wins: 47 vs 766 us at D=100, N=3e6. Branch is NOT yet O(ln N): 12.5 us (1e6) -> 47 us (3e6) once store exceeds cap. (0216)
- Add time is dominated by disk waits beyond cache. (0162, 0168, 0173, 0177, 0215)
- Add-time cliff, D=100, caps 64M/256M/1G (single runs, 0218): 5-13 us/row plateau, then a 10-100x step at N ~ 1.5e5 / 1.1e6 / 2.2e6 (2-4k rows per MB of cap; store is 5-12x the cap at the step). Writes rise first (1.4 -> 5 KB/row at 1G), reads follow (0-11 KB/row). Hot set (0219, fincore, D=100, 256M): the cache fills with tree_codes.bin (165 MiB at N=1.02e6; meta 11, slots 23). Touched code pages grow ~176 B/row (1.4-1.5 pages per 8 KB leaf region), so N*~(C-60MiB)/176B~1.1e6 matches the 256M step. At 1G the formula gives 5.5e6 but the onset was 2.2e6 (untested: dirty-page limit). A fixed-size buffer only moves the cliff (flush gives each leaf 45M/N rows); per-level buffers are needed.
- Leaf coefficient L=30, 10 paired seeds, N<=1e5 (0218): D=10 quality parity with main at every N (dloglik -0.003, SE 0.002 at 1e5, p=0.32); query 0.50x main at 1e5 but 1.8x slower at 1e4 and 2.9x at 1e3. D=100 gap remains (-0.010, p=0.007; no skill there).
- Storage-layer model, 256M cap, D=100 rows (0220; 2 buffered runs, 1 in-place): in-place appends step 14.8 -> 349 us/row between N=2.1e6 and 2.7e6. 16-way buffered flush (1024-row buffers) has no step but rises ~1.5-4 us (N<1e6) -> ~30 us (N>=1e7), then plateaus to 2e7 at ~470 B written + ~250 B read per row. Not a clean a+b lnN yet; model only (no tree, no splits). evals/meta_loop/buffer_sim.sh, fit_0220.py, fig_0220.py.

- Buffer-size sweep, storage model, 256M cap (0221; single runs): add time at N=1-2e7 is 27 us (1024-row buffers), 7 (4096), 3.8 (16384); 256 rows 72. Fanout 16 beat 8, 64. Bounded runs to N=1e8: 4096 rows 3.9->11.8 us (slope 2.1 us/lnN), 16384 rows 4.0->5.9 (0.56), no step at the cap. Cost: a query must scan the buffers on its path (~5k/17k/62k rows for 1k/4k/16k); NOT measured. evals/meta_loop/sweep_0221*.sh, long_0221.sh, fit_0221.py, fig_0221.py.

- Buffer-aware query in the model (0222; 256M cap, one run each, to N=1e8; 1024-row run stopped at 7.9e7): query reads every buffer on its path. Mean query over 1e7<N<=1e8: 1.6 ms (1024 rows), 2.7 ms (4096), 9.2 ms (16384), vs 36/112/241 us below 1e6: a 24-44x rise, not a+b lnN (slopes 191/528/3757 us per lnN). Adds repeat 0221 (30.4/9.4/5.1 us). No buffer size is good on both. evals/meta_loop/long_0222.sh, fit_0222.py, fig_0222.py (buffer_sim.py now times queries; UNCOMMITTED edit).

- Run-indexed buffers in the model (0223; 256M cap, 4096-row buffers, one run, to N=1e8): sorted runs with a header page, query = 2 preads per run. Query mean over 1e7<N<=1e8 is 10.8 ms vs 2.7 ms for the full scan (3.8x worse); add 11.3 vs 9.4 us. Cause (hypothesis): up to 16 runs per buffer x 6 levels = ~190 random reads. evals/meta_loop/long_0223.sh, fig_0223.py (fits via fit_0222.py) (buffer_sim.py `idx` mode, UNCOMMITTED).

- One sorted run per buffer in the model (0224; 256M cap, one run each, to N=1e8; HEAD b2cbcdc + UNCOMMITTED mrg edit): 4096-row buffers: query 25/705/1252 us (N<1e6 / 1e6-1e7 / 1e7-1e8) vs scan 112/1199/2731; add 1.9/11.8/24.6 vs scan 1.6/5.2/9.4 us. 1024 rows: query 1195, add 34 us at 1e7-1e8. Above N=3e7 growth is slow (query +20% / +38% from 3e7 to 1e8; add slope ~6.5 us/lnN). The 25-30x query rise at N~1e6-1e7 coincides with the cap; random 8KB reads cost 1.6 us under the cap and 250-420 us over it (4-point microbenchmark, 0224), so the step is probably cold-read latency that any disk design pays (hypothesis; unbuffered baseline not yet measured). evals/meta_loop/long_0224.sh, fig_0224.py, readfloor_0224.py.

- Leaf-only baseline in the model (0225; 256M cap, 4096-row sorted-run buffers, one run, to N=3e7, HEAD b2cbcdc + staged buffer_sim.py edit): one 8 KB leaf pread costs 1.7 us (N<1e6), 171 us (1e6-1e7), 207 us (1e7-3e7): a 100x step at N~2e6 where the leaf file exceeds the cap, then flat (9 us/lnN). Buffered query 32/605/1139 us = 4.7-5.6x the leaf-only read at 4.0-4.3 preads/query. So the post-cap step is cold-read latency any design pays; the controllable part is the 4-5x from buffer reads. evals/meta_loop/long_0225.sh, fig_0225.py.

- Five-metric scorecard, branch 84fba2d vs main 3246c4e, 1 GB cap, paired seeds (0226; 10 seeds, 5 at 1e6): branch does NOT beat main on all metrics. Add time 2.0-6.1x slower at every N (D=10, N=1e6: 7.3 vs 1.2 us/row). Query 1.3-2.4x slower at N<=1e4, 0.58-0.69x at 1e5, 0.39x at D=100 N=1e6 (1.05x at D=10 N=1e6). RSS 0.47x at D=100 N=1e6, equal at D=10. Quality identical at N<=1e4; D=100 N=1e5 significantly worse (dloglik -0.0105, p=0.007); at 1e6 dloglik ~-0.015, p>=0.18 (underpowered). evals/meta_loop/scorecard_0226.sh/.py.

## Tried and rejected
- Run-indexed buffers, header+data page per run (0223): queries 3.8x slower than full scan at 4096 rows.
- Per-child buffer segments to cut query scan 16x (0221): adds 5-10x slower in the model (44-58 us at N<=6e6 for 4096 rows).
- Moving leaf metadata into the code page: would not help; metadata is not the hot set (0219). Repacking to one page per leaf moves the cliff <2x only.
- Keeping the upper tree in memory: no help (0173).
- Dropping the prefetch entirely: slower after the cliff (0215).
- Leaf coefficient c=3: recall 0.79 at D=10, loglik worse than c=10 (0217).

## Plan (priority: O(ln N) add with no cliff; quality never worse than main)
1. Leaf coefficient: default raised to 30 in the working tree (UNCOMMITTED, one constant + one test value; 0218). 10-seed replication done for N<=1e5; still needed: N=1e6 quality and query cost vs N.
2. Replace the D=100 test function with one that has learnable structure (low-dimensional projection), so recall effects show up there. Re-check the D=100 gap, which c=30 did not close.
3. Decide whether to commit the prefetch gate (tree_codes.rs, committed in 4bdadb2; needs operator OK or a 3-run replication of the cliff).
4. (0225 DONE: leaf-only baseline measured. NEXT: keep top-level buffers in RAM, buffer only lowest 2-3 levels on disk; count cold reads per level; tiered merge to cut write amplification.) (0224: one sorted run per buffer halves query vs scan at 4096 rows but adds 2.6x slower.) (0220-0223: buffering removes the add cliff, but queries grow 24-44x with full scan (0222) and worse with run index (0223). NEXT: one run per buffer (leveled merge) and count preads per query; buffer on only the lowest 2-3 levels; require query AND add ~a+b lnN across the cap before any Rust work.) Be-tree buffered adds (ideas.md P1) + smaller touch set (P2). Bytes per add are now measured (0218, evals/meta_loop/bytes_per_add.py). The metadata-move test is WITHDRAWN (0219: meta is only 11-26 MiB). Next: log-structured code store (append code rows in arrival order to one sequential file; per-leaf id lists; gather at query or compact at split), or one buffered level. Check recall and query time afterwards. Re-measure recall afterwards. Repeat the cap sweep 3x per cap first.
5. Find the remaining 2-6 us/row in-RAM add cost (perf is blocked; use in-code timers).
Harnesses: evals/meta_loop/ (fig_0219.py; bytes_per_add.py now has FINCORE=1; bytes_per_add.py + cap_sweep.sh + fig_0218.py; leaf_sweep.sh; paired_quality.py; leaf_table.py; recall_cmp.py + recall_grid.sh; main_quality_grid.sh; quality_cmp.py; fig_0217.py; fig_0216.py; scaling_capped.sh), scripts/bench_cmp/bench.py.

## Questions for the operator
- (0225) The cold-read step is ~100x for even one leaf read (1.7 -> ~200 us). Will you accept "a one-time step of cold-read size, then a+b lnN with a close to the one-read floor" as meeting VISION's no-step goal? Also answers the 0224 question.
- VISION asks for no step at the cap, but a random read costs ~250-420 us over the cap vs ~2 us under it on this disk (0224). Is 'bounded step, then a+b lnN' acceptable for queries? (0224)
- (0226) Beating main on all five metrics is now measured and unmet (add time worst). Is add-time parity with main in RAM required, or only past the cliff? Should 1e6 quality be rerun with 10 seeds?
- VISION-b.md does not exist (only VISION.md, which I used). Is another file intended?
- Default leaf coefficient 10 -> 30: I applied it in the working tree (uncommitted) because VISION.md says quality is never worse than today. It triples leaves searched per query, keeps O(ln N), closes the D=10 gap, but makes queries 1.8x slower than main at N=1e4, D=10. Keep it? (0218)
- Is a test function with learnable structure at D=100 acceptable for judging quality? The current one gives no skill there at N<=1e5. (0217)
- May the prefetch gate be committed?
- Is "beat main on add time" required in RAM too, where main only appends and the branch inserts into a tree? Or only at scale / past the cliff?
- Is a log-structured code store (rows gathered by id at query time) acceptable if it costs some query time at small N? (0219)
- Is a 2-5 us/row-per-decade smooth rise at 4k-16k buffers acceptable if queries must scan 17k-62k buffered rows? What query-time budget per buffered row is allowed? (0221)
- Is a ~20x smooth rise (1024-row buffers, 0220) over one decade of N (0220 model) acceptable evidence of progress, or must the target a+b lnN fit hold across the cap crossing before any Rust work? (0220)
- Is the model query (full read of path buffers) a fair stand-in, or should the model use an optimised reader first? Decision made without you: I judged it fair as an upper bound and moved on. (0222)
- Should I stop modelling buffers and instead try a design where queries never touch buffers (e.g. insert visible only after flush, with a small in-RAM recent-rows list)? It relaxes read-your-writes. (0223)
