# STATE (edit in place; reports are the log)

## Established (1 GB cgroup cap unless noted; "branch" = dsweet/agg_obs HEAD 7de1ea7)
- Queries: branch faster than main at N>=1e5 (D=10: 0.22 vs 0.89 s; D=100: 2.2 vs 3.2 s; 1000 queries); at N<=1e3 main faster. (0216; 0215-progress report: 480x at D=1000, N=1e5, 4 GiB cap.)
- Memory: branch RssAnon 21 MB vs main 558 MB at D=100, N=1e6 (0216).
- Quality (3 paired seeds, D=10/100, N up to 1e5): branch loglik -0.025 (sd 0.024) / -0.008 and NRMSE +0.002 / +0.006 vs main at N=1e5. Not significant, sign favours main. Identical at N<=1e3. (0216)
- Add time: main is 2.5-3x faster in RAM (gate build: 2.0 vs 0.8 us/row at D=10, N=1e5). Past the cliff the branch wins: 47 us vs 766 us at D=100, N=3e6. Branch is NOT yet O(ln N): 12.5 us (1e6) -> 47 us (3e6) once store (6.5 GiB) exceeds the cap. (0216)
- Add time is dominated by disk waits (dirty-page throttling, mmap/munmap) beyond cache (0162, 0168, 0173, 0177, 0215).
- Once-per-page prefetch gate (tree_codes.rs, UNCOMMITTED): in-RAM add 2-3x faster than head, quality bit-identical, 181 bpann tests pass; cliff only ~15% better (single run). (0216)

## Tried and rejected
- Keeping the upper tree in memory: no help (0173).
- Dropping the prefetch entirely: slower after the cliff (0215).

## Plan (priority: add time must follow a + b ln N, no cliff)
1. Decide whether to commit the prefetch gate (needs operator OK or a 3-run replication).
2. Be-tree buffered adds (ideas.md P1) + smaller touch set (P2); first measure bytes written per add above the cap.
3. Quality gap: 10 seeds + neighbour recall of tree vs exact search; if recall explains it, adapt leaf budget.
4. Find the remaining 2-6 us/row in-RAM add cost (perf is blocked; use in-code timers).
Harnesses: scripts/bench_cmp/bench.py and evals/meta_loop/ (quality_cmp.py, quality_grid.sh, scaling_capped.sh, fig_0216.py).

## Questions for the operator
- VISION-b.md does not exist; I used VISION.md. Is another file intended?
- Is "beat main on add time" required in RAM too, where main only appends and the branch inserts into a tree? Or only at scale / past the cliff?
- Is a loglik deficit of ~0.01-0.03 vs main acceptable if queries are 3-4x faster?
- May the prefetch gate be committed?
