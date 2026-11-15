# STATE (edit in place; reports are the log)

## Established (1 GB cgroup cap unless noted; "branch" = dsweet/agg_obs, HEAD 61ed15c (prefetch gate committed 4bdadb2); "main" = 3246c4e)
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

- Where in-RAM add time goes (0233, timer build, 1 run, sandbox, N=1e5): insert_row 2.6 us/row (D=10), 5.7 (D=100). Splits 34%/45% (writes to old/new leaf dominate, alloc_near 0.23 us/row, median_split 0.1-0.2), descent 22%/16%, leaf push 10%/23%, morph 0.04. Shares only; timers inflate totals.

## Tried and rejected
- File fragmentation (0230), writeback contention (0229), 8K alignment (0229 data) as causes of slow buffer-file reads.
- Smaller fence granularity (16 vs 64 rows): only -10% query (0228).
- Fixed number of disk levels with RAM upper buffers: query unchanged, RAM grows with N (0227).
- Run-indexed buffers: queries 3.8x slower than full scan (0223). Per-child buffer segments: adds 5-10x slower (0221).
- Moving leaf metadata into the code page; one page per leaf (0219). Keeping the upper tree in memory (0173). Dropping the prefetch (0215).
- Leaf coefficient c=3: recall 0.79 at D=10 (0217).
- Wider AVX2 loop in l2_sq_f32 for D<32 (0233): no gain (1.87 vs 1.66 us/row insert).
- Morph work as the cause of the in-RAM add gap (0232).

## Plan (priority: O(ln N) add and query with no cliff; quality never worse than main)
0. MODEL STOPPED at 0231 (decision without operator: further model runs cannot change the choice).
1. Rust, behind a flag: level-1 buffer inside the leaf page (0228), so one pread serves both. Acceptance: query and add fit a+b lnN across the cap (3 runs per cap), then scorecard_0226 with 10 seeds.
2. Leaf coefficient 30 (uncommitted in working tree; 0218): replicate N=1e6 quality, query cost vs N.
3. Learnable D=100 test function (low-dimensional projection) to judge recall.
4. Gate done (0232, uncommitted: commit if operator agrees). Profiled in 0233 (splits dominate). Next: chunked fallocate of slot file / minor-fault count per row to test whether fresh-page writes drive split cost; repeat capped runs 3x with SE. Rust leaf-overflow layout (plan 1) NOT started; still the priority for O(ln N). Harness: add_phase_prof.sh/.patch.
Harnesses: evals/meta_loop/ (add_gate_cmp.sh, advise_ab.sh, fig_0232.py, scorecard_0226.*, paired_quality.py, buffer_sim.*, long_02xx.sh, fig_02xx.py, bytes_per_add.py, cap_sweep.sh, leaf_sweep.sh, recall_cmp.py), scripts/bench_cmp/bench.py.

## Questions for the operator
- Approve the leaf-overflow layout change in Rust (0228-0231)? Asked repeatedly with no answer; I will implement it behind a flag unless told otherwise.
- Is "bounded one-time step of cold-read size, then a+b lnN" acceptable as meeting VISION's no-step goal for queries? (0224, 0225)
- Must add-time parity with main hold in RAM, or only past the cliff? (0226) Rerun 1e6 quality with 10 seeds?
- Keep leaf coefficient 30 (closes D=10 gap, queries 1.8x slower than main at N=1e4)? (0218)
- Is a learnable-structure test function at D=100 acceptable for quality? (0217)
- Is a log-structured code store, or queries that skip buffers (relaxed read-your-writes), acceptable? (0219, 0223)
- Commit the AdviseGate change (sync.rs, 0232)? Thresholds (8 faults/32 adds, 32 groups) are untuned.
- Sandbox lacks perf (paranoid=4), gdb, valgrind; is a profiling-capable environment available? (0233)
- VISION-b.md does not exist; is another file intended?
