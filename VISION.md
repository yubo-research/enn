

# Vision
- Python exposes the public API to Python users, but all the work is in Rust.
- The running time of ENN adds and queries (posterior, posterior_function_draw) grows as k*D*lnN (or better) (k is number of neighbors, D is number of dimensions, N is number of observations). Measuring the scaling is crucial. Lower wall time is nice, but it doesn't equate to good scaling. O(lnN) must hold from N=1 to N so large that the dataset cannot fit in RAM. Success: add and query time per operation, measured under the cap C, follows a + b·lnN from N=1 to N whose data is ≥10× C, with no step at the point where the data outgrows C.
- We are designing a disk-backed-storage ANN algorithm. Store the data on disk from N=1 onward.
  - DON'T build a two-mode system where you store the data in memory when N is small.
  - DON'T build a two-mode system where you do a linear scan when N is small and a log-time search when N is large.
  - Build a single, coherent disk-based, log-time algorithm that works for all N.
  - Use memory to decrease wall time by adapting to access patterns.
  - Avoid any "big bang" or "stop-the-world" events during normal use of ENN.

- Maximum memory is 1GB, independent of N (and of everything else). This includes page cache: everything the kernel charges to the process's cgroup counts. The test process is also limited to 1 CPU. Test under `systemd-run --user --scope -p MemoryMax=C -p MemoryHigh=C -p CPUQuota=100%` with the store on a real disk (not `/tmp`); C=256MB is an acceptable scale model, C=1GB is the target. Our disk use should scale with N and may grow to any size, in principle.
  - It may be tempting to design an algorithm that just puts all data in memory until 1gb is hit and then "switches gears" to a different algorithm that uses disk and it much slower. Resist temptation. Use the disk right from the start, and use the memory to speed help make the algorithm more efficient so that there is no performance "cliff" when the user adds enough data s.t. the 1GB limit is reached. Query time (also add time) vs N should be smooth right through the 1GB limit. You should construct a single algorithm that works one way from N=1 to N->infinity.
- Prediction quality (loglik, rmse) is never worse than it is today. See ./ops/evaluate.py for quality metrics. Use standard errors and good statistical tests to determine whether prediction quality changes are significant. We might accept big speed improvements in exchange for statistically insignificant prediction degredation. Use your own judgement.
- Adds and queries can't be too fast (constrained by good prediction quality).
  - Approximations are acceptable, as long as prediction quality is not sacrificed.
- We should always do calculations in one thread. Housekeeping may be backgrounded to a second thread, but our target system has a single core.
- Since BPANN_DISK is as on-disk data store, maybe we can learn about efficiency and speed from database or filesystem literature or code.
- Consider using or adding to the set of evals (see evaluate.py) so that we have clear, repeatable measures of time complexity, memory complexity and prediction quality.

# Misc
- It is important that metric learning continues to function efficiently (lnN, constant memory whether metric learning is on or off).
- It would be interesting to set k, the number of neighbors, automatically based on summary statistics, fitting, or heuristics. It seems intuitive that k should be larger for noisier data and larger when data is more densely packed about the query point, but (i) we don't know the noise level a priori, and increasing k with N could ruin the O(k lnN) time complexity. Leaving k as a constant specified by the user isn't terrible. It is an open question whether we could find k automatically.


Possibly-helpful refs
- B+ANN: https://arxiv.org/pdf/2511.15557
- e-means clustering: https://arxiv.org/pdf/2611.09501
- TurboQuant: https://arxiv.org/pdf/2504.19874
- ENN: https://arxiv.org/pdf/2506.12818



# Primary concern right now
Time and memory complexity up to very large N. Like N = 1e9 X D = 10000.  (This is a huge dataset, though, snd wouldn't fit on the disk we have. Also, it would take a very long time to run. Still, do your best to analyze and evaluate. Maybe in the future we'll procure resources to test larger problems.)  The key point is to focus first on designing a scalable, generalizable, principled algorithm and to avoid overfitting to the machine or to a specific problem size; and second on good engineering for speed.
