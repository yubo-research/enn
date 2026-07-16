# Plan: Add `draw` command to ops/stress.py

## User Request

New command in `ops/stress.py` called `draw` that should:
- Draw a sample of NUM_OBS points, x, uniformly in [0,1]^num_dim
- Set y = f(x) + 0.1 * N(0,1) (iid noise at every observation)
- Use f(x) = (x - 0.3)**2
- Fit the hyperparameters, including aleatoric variance
- Draw a new sample of NUM_TEST points, x_test, uniformly in [0,1]^num_dim
- Compute and report the average likelihood over x_test, using y_test = f(x_test) + 0.1*N(0,1)
- Use observation_noise=True for all operations

## Current State

### ops/stress.py CLI

- Click group `cli` with two subcommands:
  - `enn` — stream synthetic adds, time queries at checkpoints
  - `sample` — reopen a disk bpann store and draw posterior function samples
- Shared helpers: `make_uniform_query_points` (uniform in `[low, high]`, defaults `[-1, 1]`), `DEFAULT_NUM_DIM = 10`, `STRESS_QUERY_SEED = 1`
- No synthetic regression / fit / test-likelihood path today

### Fitting and likelihood (library)

| Piece | Location | Behavior |
| --- | --- | --- |
| `enn_fit` / `ENNStatefulFitter` | `src/enn/enn/enn_fit.py`, `enn_fitter.py` | Random-search hyperparams; `infer_aleatoric_variance_scale=True` by default |
| `subsample_loglik` (Rust) | `rust/crates/ennbo/src/fit.rs` | Already evaluates candidates with `PosteriorFlags { exclude_nearest: true, observation_noise: true }` |
| TuRBO-ENN defaults | `rust/crates/ennbo/src/config.rs` `turbo_enn_config` | `k=10`, `num_fit_candidates=30`, `num_fit_samples=10` |
| Posterior API | `EpistemicNearestNeighbors.posterior(..., flags=PosteriorFlags(...))` | `observation_noise` defaults False; must pass True for test eval |

### Adjacent patterns

- Multi-dim squared-distance reduction in-repo (sign differs; used as maximizer objective): `-np.sum((x - 0.3) ** 2, axis=1)` in `tests/parity/test_optimizer_coverage_gaps.py`
- Stress tests: `tests/test_ops_stress.py`, `tests/test_ops_stress_sample.py` (CliRunner + small helper coverage; `sample` marked slow)

## Requested Changes

1. Add a `draw` Click subcommand on `ops/stress.py` that builds a synthetic train set on `[0,1]^num_dim`, fits ENN hyperparameters (including aleatoric), evaluates average predictive likelihood on a fresh noisy test set, and prints the result.
2. Use `observation_noise=True` for fit evaluation (already true inside `subsample_loglik`) and for the test-set posterior used to score likelihood.
3. Cover the new path with unit/CLI tests mirroring existing stress command tests.

## Q&A

### Q1. How is `f(x) = (x - 0.3)**2` defined when `num_dim > 1`?

**Answer:** Reduce over dimensions: `y_i = sum_j (x_{ij} - 0.3)^2`, shape `(n, 1)`. Same reduction as in `tests/parity/test_optimizer_coverage_gaps.py` (that file negates the sum for maximization; here we keep the positive sum as `f`).

### Q2. Does “average likelihood” mean mean PDF or mean log-likelihood?

**Answer:** Mean of per-point Gaussian predictive densities \(p(y_i \mid x_i) = \mathcal{N}(y_i; \mu_i, \mathrm{se}_i^2)\) under the fitted posterior with `observation_noise=True`, then `mean` over the `NUM_TEST` points. Do **not** reuse `subsample_loglik` for the reported metric (it returns a sum of y-scaled log-likelihoods with `exclude_nearest=True`, not average likelihood).

### Q3. What CLI shape and defaults?

**Answer:** Required positional args `NUM_OBS` and `NUM_TEST`; `--num-dim` is an option defaulting to `DEFAULT_NUM_DIM` (10).

```text
./ops/stress.py draw NUM_OBS NUM_TEST [--num-dim N] [--seed S] [--k K]
  [--num-fit-candidates C] [--num-fit-samples P]
```

Other defaults: `seed=0`, `k=10`, `num_fit_candidates=30`, `num_fit_samples=10` (TuRBO-ENN defaults). In-memory `FLAT` index only (no `--work-dir`); this is a fit/eval experiment, not a disk stress path.

### Q4. Does fitting already honor `observation_noise=True`?

**Answer:** Yes. `subsample_loglik` hardcodes `.with_observation_noise(true)` (`rust/crates/ennbo/src/fit.rs`). The new command must still pass `PosteriorFlags(observation_noise=True)` when computing the test posterior for the reported average likelihood. Aleatoric inference comes from `ENNStatefulFitter(infer_aleatoric_variance_scale=True)` (the default used inside batch `enn_fit`); `enn_fit` itself has no such kwarg.

### Q5. How should train/test/fit RNGs relate?

**Answer:** `--seed` drives data only: `data_rng = np.random.default_rng(seed)`. Draw train `x`/`noise`, then test `x_test`/`noise` sequentially from `data_rng`. Fit uses a separate RNG: `fit_rng = np.random.default_rng(seed + 1)`. Do **not** pass `data_rng` into `enn_fit`.

### Q6. How to sample uniform `x` in `[0,1]^num_dim`?

**Answer:** Inline `rng.uniform(0.0, 1.0, size=(n, num_dim))` inside `make_draw_observations`. Do **not** use `make_uniform_query_points` (it takes a seed and builds its own RNG, which fights the shared `data_rng` stream).

## Plan

### Phase 1 — Implement `draw` in ops/stress.py

- [x] Add helpers (module-level, next to existing synthetic helpers):
  - `draw_f(x) -> np.ndarray` — `np.sum((x - 0.3) ** 2, axis=1, keepdims=True)`
  - `make_draw_observations(num_obs, *, num_dim, rng)` — `x = rng.uniform(0.0, 1.0, size=(n, num_dim))`, then `y = draw_f(x) + 0.1 * rng.standard_normal((n, 1))`
  - `gaussian_likelihood(y, mu, se) -> np.ndarray` — per-point \(\mathcal{N}(y;\mu,\mathrm{se}^2)\) densities
  - `average_likelihood(y, mu, se) -> float` — `float(np.mean(...))`
- [x] Add `DrawStressConfig` / `DrawStressResult` dataclasses (num_obs, num_test, num_dim, seed, k, fit knobs, fitted scales, avg_likelihood, timings as useful).
- [x] Implement `run_draw_stress(config) -> DrawStressResult`:
  1. `data_rng = default_rng(seed)`; `fit_rng = default_rng(seed + 1)`
  2. Build train `(x, y)` with `make_draw_observations(..., rng=data_rng)`
  3. Build test `(x_test, y_test)` the same way (still `data_rng`)
  4. `EpistemicNearestNeighbors(x, y, scale_x=False)` (FLAT default)
  5. `enn_fit(..., k=..., num_fit_candidates=..., num_fit_samples=..., rng=fit_rng)` (aleatoric inferred via fitter default)
  6. `posterior(x_test, params=fitted, flags=PosteriorFlags(observation_noise=True))`
  7. Compute and store average likelihood
- [x] Add Click command `draw` with required args `num_obs`, `num_test`; option `--num-dim` default `DEFAULT_NUM_DIM` (10); plus `--seed` / `--k` / fit-knob options above; validate `>= 1` for sizes/dims and fit knobs; echo a config header line and a summary line that includes fitted `epistemic_variance_scale`, `aleatoric_variance_scale`, and `avg_likelihood=...` (same two-line style as `sample`).

**Validation:** `python -c "from ops.stress import cli; ..."` / CliRunner invoke `draw 40 20 --num-dim 2 --seed 0` exits 0; output contains `avg_likelihood=` and a finite float; fitted `aleatoric_variance_scale >= 0`.

### Phase 2 — Tests

- [x] Add `tests/test_ops_stress_draw.py`:
  - `draw_f` shape/value checks (1d and multi-d)
  - `make_draw_observations` bounds in `[0,1]`, reproducibility for fixed seed
  - `run_draw_stress` on small `num_obs`/`num_test` returns finite `avg_likelihood` and positive epistemic scale
  - CliRunner: happy path + rejects `num_obs < 1` / missing args
- [x] Keep tests fast (small n/d, modest fit candidates/samples); mark slow only if runtime warrants it (prefer not).

**Validation:** `pytest tests/test_ops_stress_draw.py -q` passes; `pre-commit`/existing stress tests still pass.
