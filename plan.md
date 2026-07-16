# Plan: Add argmin-RMS metric to `ops/stress.py draw`

## User Request

Add a second metric to the `draw` command that measures joint-sample quality via the path functional \(g(y)=\arg\min_x y(x)\):

- Draw `NUM_TEST` shared \(x\) values.
- Draw \(y(x)\) from the posterior (both `posterior().sample` and `posterior_function_draw`).
- For this metric use `observation_noise=False`.
- \(\varepsilon = \hat{x} - 0.3\cdot\mathbf{1}_d\) where \(\hat{x}=g(y)\) and true \(f(x)=\sum_j(x_j-0.3)^2\).
- Metric: \(\mathrm{RMS}(\varepsilon)=\sqrt{\mathrm{mean}_s\|\varepsilon^{(s)}\|_2^2}\) over many draws (on the order of 100).
- Keep marginal avg likelihood as the first metric; agree that the same candidate \(x\)'s are used for both methods, multi-dim uses \(\ell_2\) residual, and the discrete-grid caveat applies.

## Current State

### `draw` command (`ops/stress.py`)

- Fits ENN on synthetic train data (`draw_f` + noise); scores two methods on the same `x_test`:
  1. `posterior` — `posterior(..., flags=DRAW_FLAGS)` then `.sample()`; `avg_likelihood` from analytic \(N(y;\mu,\mathrm{se}^2)\).
  2. `posterior_function_draw` — joint draws; `avg_likelihood` from empirical mean/std over sample axis.
- `DRAW_FLAGS = PosteriorFlags(observation_noise=True)` used for **both** methods today.
- Draw tensors are `(batch, metrics, num_samples)` for both APIs (Python `posterior_function_draw` transposed in `_finalize_function_draw`).
- CLI: `./ops/stress.py draw NUM_OBS NUM_TEST [--num-dim] [--seed] [--k] [--num-fit-candidates] [--num-fit-samples] [--num-samples]`.
- Defaults: `DEFAULT_DRAW_NUM_SAMPLES = 10`; stdout is header + two method lines with `avg_likelihood=`, `draws_shape=`, `all_finite=`, `eval_s=`.

### Synthetic objective

- `draw_f(x) = sum_j (x_j - 0.3)^2` → true minimizer \(x^\star = 0.3\cdot\mathbf{1}_d\) on \([0,1]^d\).
- Center `0.3` is inline in `draw_f` only (no named constant).

### Tests

- `tests/test_ops_stress_draw.py` asserts finite `avg_likelihood` and CLI lines starting with `posterior avg_likelihood=` / `posterior_function_draw avg_likelihood=`.
- No path-functional / argmin metric coverage yet.

## Requested Changes

1. Keep marginal `avg_likelihood` scoring with `observation_noise=True` (unchanged intent).
2. Add `argmin_rms` on both method reports: discrete \(\arg\min\) over the shared `x_test` grid of each draw’s metric-0 values, residual to \(x^\star=0.3\cdot\mathbf{1}\), RMS of \(\ell_2\) residuals over draws.
3. Compute `argmin_rms` draws with `observation_noise=False` for both `posterior().sample` and `posterior_function_draw`.
4. Use the same `x_test` for both methods and both metrics.
5. Raise default `--num-samples` to 100 (enough Monte Carlo for RMS).
6. Update CLI summary lines and tests accordingly.

## Q&A

### Q1. Same `--num-samples` for likelihood draws and argmin-RMS draws?

**Answer:** Yes. One `num_samples` drives all Monte Carlo counts. Default becomes `100`. Likelihood for `posterior` remains analytic (does not require samples); `posterior().sample` and both `posterior_function_draw` passes still use `num_samples`.

### Q2. Separate posterior calls for `observation_noise=True` vs `False`?

**Answer:** Yes. Keep `DRAW_FLAGS` (`observation_noise=True`) for avg-likelihood paths. Add `DRAW_FLAGS_NO_OBS = PosteriorFlags(observation_noise=False)` for argmin-RMS draws only. That means an extra `posterior(...).sample` and an extra `posterior_function_draw` under the no-obs flag (four draw/eval paths total: lik×2 + rms×2, with posterior lik still analytic).

### Q3. How is multi-dim \(\varepsilon\) defined?

**Answer:** For draw \(s\), \(\hat{x}^{(s)} = x_{\mathrm{test}}[i^\star]\) with \(i^\star=\arg\min_i y^{(s)}_i\) (metric 0). \(\varepsilon^{(s)}=\hat{x}^{(s)}-0.3\cdot\mathbf{1}_d\). \(\mathrm{RMS}=\sqrt{\frac{1}{S}\sum_s\|\varepsilon^{(s)}\|_2^2}\). Introduce `DRAW_F_CENTER = 0.3` and use it in `draw_f` and the RMS helper.

### Q4. Discrete-grid limitation?

**Answer:** Accepted. Metric only searches the finite `x_test` set; high-\(d\) random grids rarely near \(x^\star\). Tests keep small `num_dim` (e.g. 2). No change to how `x_test` is generated.

### Q5. Which draw tensor is reported in `draws_shape` / `all_finite`?

**Answer:** Report shapes/finiteness from the **argmin-RMS** draws (`observation_noise=False`), since those are the joint-quality draws. `avg_likelihood` stays on the ON=True path as today.

## Plan

### Phase 1 — Argmin-RMS helpers and dual-flag scoring in `run_draw_stress`

- [x] Add `DRAW_F_CENTER = 0.3`; use it in `draw_f`.
- [x] Add `DRAW_FLAGS_NO_OBS = PosteriorFlags(observation_noise=False)`.
- [x] Set `DEFAULT_DRAW_NUM_SAMPLES = 100`.
- [x] Add `argmin_rms(x_test, draws) -> float`:
  - `draws` shape `(B, M, S)`; use metric 0.
  - For each sample \(s\): `i = argmin(draws[:, 0, s])`; `eps = x_test[i] - DRAW_F_CENTER`; accumulate `||eps||_2^2`.
  - Return `sqrt(mean)`.
- [x] Extend `DrawMethodResult` with `argmin_rms: float`.
- [x] In `run_draw_stress`:
  1. Fit as today.
  2. **Likelihood (ON=True):** existing `posterior` analytic lik; existing `posterior_function_draw` empirical lik.
  3. **Argmin RMS (ON=False):** `posterior(...).sample` and `posterior_function_draw` with `DRAW_FLAGS_NO_OBS` on the same `x_test`; compute `argmin_rms` for each.
  4. Fill each `DrawMethodResult` with both metrics; `draws_shape` / `all_finite` / method `eval_s` from the ON=False draw pass (time both passes into `eval_s` or only the RMS pass — use wall time covering that method’s lik+rms work for a single `eval_s`).
- [x] Update `format_draw_method_summary` to include `argmin_rms=...`.

**Validation:** `./ops/stress.py draw 40 20 --num-dim 2 --seed 0 --k 5 --num-fit-candidates 8 --num-fit-samples 5 --num-samples 20` prints three lines; both method lines contain `avg_likelihood=` and `argmin_rms=` with finite values; `draws_shape` matches `(20, 1, 20)` on both lines.

### Phase 2 — Tests

- [x] Unit-test `argmin_rms`: hand-built `x_test` / `draws` where the min is known → expected RMS.
- [x] `run_draw_stress`: assert finite `argmin_rms` for both methods; assert `posterior_function_draw.argmin_rms <= posterior.argmin_rms` is **not** required (stochastic), only finiteness and `>= 0`.
- [x] CLI test: both method lines contain `argmin_rms=`; default `num_samples` in help/header reflects 100 when not overridden.
- [x] Keep existing likelihood / rejection tests green.

**Validation:** `pytest tests/test_ops_stress_draw.py -q` passes.
