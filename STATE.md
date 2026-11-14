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

## Tried and rejected
- Moving leaf metadata into the code page: would not help; metadata is not the hot set (0219). Repacking to one page per leaf moves the cliff <2x only.
- Keeping the upper tree in memory: no help (0173).
- Dropping the prefetch entirely: slower after the cliff (0215).
- Leaf coefficient c=3: recall 0.79 at D=10, loglik worse than c=10 (0217).

## Plan (priority: O(ln N) add with no cliff; quality never worse than main)
1. Leaf coefficient: default raised to 30 in the working tree (UNCOMMITTED, one constant + one test value; 0218). 10-seed replication done for N<=1e5; still needed: N=1e6 quality and query cost vs N.
2. Replace the D=100 test function with one that has learnable structure (low-dimensional projection), so recall effects show up there. Re-check the D=100 gap, which c=30 did not close.
3. Decide whether to commit the prefetch gate (tree_codes.rs, committed in 4bdadb2; needs operator OK or a 3-run replication of the cliff).
4. Be-tree buffered adds (ideas.md P1) + smaller touch set (P2). Bytes per add are now measured (0218, evals/meta_loop/bytes_per_add.py). The metadata-move test is WITHDRAWN (0219: meta is only 11-26 MiB). Next: log-structured code store (append code rows in arrival order to one sequential file; per-leaf id lists; gather at query or compact at split), or one buffered level. Check recall and query time afterwards. Re-measure recall afterwards. Repeat the cap sweep 3x per cap first.
5. Find the remaining 2-6 us/row in-RAM add cost (perf is blocked; use in-code timers).
Harnesses: evals/meta_loop/ (fig_0219.py; bytes_per_add.py now has FINCORE=1; bytes_per_add.py + cap_sweep.sh + fig_0218.py; leaf_sweep.sh; paired_quality.py; leaf_table.py; recall_cmp.py + recall_grid.sh; main_quality_grid.sh; quality_cmp.py; fig_0217.py; fig_0216.py; scaling_capped.sh), scripts/bench_cmp/bench.py.

## Questions for the operator
- VISION-b.md does not exist (only VISION.md, which I used). Is another file intended?
- Default leaf coefficient 10 -> 30: I applied it in the working tree (uncommitted) because VISION.md says quality is never worse than today. It triples leaves searched per query, keeps O(ln N), closes the D=10 gap, but makes queries 1.8x slower than main at N=1e4, D=10. Keep it? (0218)
- Is a test function with learnable structure at D=100 acceptable for judging quality? The current one gives no skill there at N<=1e5. (0217)
- May the prefetch gate be committed?
- Is "beat main on add time" required in RAM too, where main only appends and the branch inserts into a tree? Or only at scale / past the cliff?
- Is a log-structured code store (rows gathered by id at query time) acceptable if it costs some query time at small N? (0219)
