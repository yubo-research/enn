# Plan: Optional per-metric y bounds (output warp)

## User Request

For approach 2 (built-in output warp): the user of the Rust or Python API should be able to optionally specify a lower bound and/or an upper bound on each y. Default bounds are `(-inf, inf)`. The library should handle all transformations transparently. The user should send natural-unit y’s to the library, and should receive posterior draws, mus, ses, etc. in natural units.
/
## Current State

### x bounds (pattern to mirror)

- Optimizer takes `bounds: (num_dim, 2)` at construction.
- Python `RustOptimizer.ask` / `tell` convert natural ↔ unit via affine map; Rust TR/acq stay in unit box (`src/enn/turbo/rust_optimizer.py`, `rust/crates/ennbo/src/candidates.rs`).
- There is **no** analogous y transform today.

### y ingress / storage

- `tell` / `add_observations` / `EpistemicNearestNeighbors::{new,add}` store **raw** y (`model.rs` → `backend.append_rows`).
- Disk: `train_y.bin` (+ optional `train_yvar.bin`); `metadata.json` records `num_dim`, `num_metrics`, `scale_x`, etc. — **no y bounds** (`backend/disk_observation.rs::write_metadata`).
- ENN does **not** center stored y. Per-metric `y_scale` is empirical std from running moments (`scale_from_moments` in `model.rs`), used only to scale SE and to divide `yvar` in weight math (`posterior.rs`).
- `standardize_y` (median + RMS in `util.rs`) is a separate util / GP path — **not** used for ENN storage.

### Posterior egress

- `mu` = weighted average of **stored** neighbor y (already “natural” only because storage is natural).
- `se` / `se_epi` / `se_ale` = √var × `y_scale[m]` (`se_from_variance_components`).
- Draws: `posterior_function_draw` / `ENNSurrogate::sample` build `mu + scaled noise` in the same space as stored y (`posterior/draw_compute.rs`).
- Python: `EpistemicNearestNeighbors.posterior` → `ENNNormal(mu, se, …)` (`enn_class.py`, `enn_normal.py`); `ENNNormal.sample` does `mu + se * eps` in whatever units `mu`/`se` have.

### Acquisition / incumbent (internal)

- UCB: `mu + β·se` on surrogate `predict` (`strategy/mod.rs`).
- Thompson: `surrogate.sample` then argmax.
- Incumbent / Morbo ranges: operate on stored y / predict mu (`incumbent_tracker.rs`, `tell_common`).
- All of this is in **stored-y units**. Affine-y invariance tests encode that (`tests/test_turbo_invariance.py`, `tests/test_turbo_adversarial.py`).

### Config surfaces

| Layer | Where | y-bounds today |
|---|---|---|
| Rust `ENNSurrogateConfig` | `surrogate.rs` | none |
| Rust `EpistemicNearestNeighbors` | `model.rs` | none |
| Python `ENNSurrogateConfig` | `enn_surrogate_config.py` | none |
| Python `EpistemicNearestNeighbors` | `enn_class.py` | none |
| `create_optimizer` / `ConfigOverrides` | `py_optimizer.rs`, `config.rs` | none |

### Multi-metric

- `num_metrics = y.ncols()`; `y_scale` is per-column. Bounds must be per-metric `(num_metrics, 2)`.

### Adjacent / out of scope today

- GP `turbo_one` path has its own `standardize_y` / `_unstandardize` (`gp_surrogate.py`) — separate from ENN.
- Direct observation accessors (`y_obs`, `observations_y`, `train_rows_at`, `_train_y`) currently return stored y.

## Requested Changes

1. Allow optional per-metric lower and/or upper y bounds on the Rust and Python ENN / TuRBO-ENN APIs; default `(-∞, +∞)` (identity).
2. Warp y (and Jacobian-scale `yvar`) on ingress; keep fit, `y_scale`, acquisition, and incumbent in warped space.
3. Inverse-warp all user-facing posterior outputs (`mu`, `se`, `se_epi`, `se_ale`, draws) and observation reads (`y_obs` / `train_rows_at` y / `_train_y`) back to natural units.
4. Reject or otherwise hard-fail observations outside the open interval implied by the bounds; document boundary ε policy for values at the finite endpoints.

## Q&A

### Q1. Which warp for one-sided vs two-sided bounds?

**Answer:** Use a transform that is **strictly increasing in y** so maximization / UCB / incumbent argmax in warped space matches natural space:

| Bounds | Warp `z = φ(y)` | Inverse |
|---|---|---|
| `(-∞, +∞)` | `y` | identity |
| `(a, +∞)` | `log(y - a)` | `a + exp(z)` |
| `(-∞, b)` | `-log(b - y)` | `b - exp(-z)` |
| `(a, b)` | `logit((y-a)/(b-a))` | `a + (b-a)·σ(z)` |

Implement as one module (e.g. `rust/crates/ennbo/src/y_bounds.rs`) with per-column bound pairs; `±∞` selects the matching branch.

### Q2. Store warped or natural y on disk / in memory?

**Answer:** Store **warped** y (and warped `yvar`) so `y_scale`, fit, and neighbor averages need no special cases. Persist `y_bounds` in `metadata.json` (shape `(num_metrics, 2)`, JSON null or `±Infinity` for open sides) and require reopen to load the same bounds. Public `train_rows_at` / `y_obs` / `_train_y` **inverse-warp** on read so callers still see natural units.

### Q3. How to map `mu` / `se` / draws to natural units?

**Answer:**

- **Draws:** sample in warped space, then elementwise `φ⁻¹` (draws stay inside `(a,b)`).
- **`mu`:** `μ_nat = φ⁻¹(μ_z)` (median under Gaussian-in-z for monotone φ).
- **`se` / `se_epi` / `se_ale`:** delta method `se_nat = |dφ⁻¹/dz|(μ_z) · se_z` (same factor on epi/ale).
- **`ENNNormal.sample`:** when bounds are non-identity, sample via `φ⁻¹(φ(μ_nat) + (se_nat / |dφ⁻¹/dz|) · ε)` (recover latent Gaussian), not raw `μ + se·ε`, so draws respect bounds. Identity bounds keep today’s `μ + se·ε`.

### Q4. Where do bounds attach in the API?

**Answer:**

- **Model:** `EpistemicNearestNeighbors::new*` / Python `__init__` take `y_bounds: Option<(num_metrics, 2)>` (None ⇒ all `(-∞,+∞)`).
- **Optimizer:** `ENNSurrogateConfig.y_bounds` (Rust + Python); passed into model construction in `ENNSurrogate::construct_model`. Shape must match `num_metrics` once y width is known (validate on first `tell` / `new` with data).
- **GP / `turbo_one`:** out of scope for this work (no change).

### Q5. Boundary / validation policy?

**Answer:** Finite bounds define an **open** interval. On ingress: if `y <= a` or `y >= b` (for finite sides), return `ENNError::InvalidParameter` (Python `ValueError`). Do not silent-clamp. Require `a < b` when both finite. `yvar` must be `>= 0` as today; after warp, `yvar_z = (φ'(y))² · yvar_y`.

### Q6. Do affine-y invariance tests still hold?

**Answer:** Yes for default unbounded bounds (behavior unchanged). Nonlinear warps break affine invariance; do **not** change those tests’ default path. Add separate bounded-y tests (round-trip, in-bounds draws, reject OOB, ask/tell with `[0,1]` metric).

### Q7. Conditional / what-if APIs?

**Answer:** `conditional_posterior(..., y_whatif, ...)` warps `y_whatif` on ingress (same as `add`). Returned posterior is naturalized on egress like `posterior`.

## Plan

### Phase 1 — Warp primitives + model I/O

- [ ] Add `y_bounds` module in `ennbo`: validate pairs; `warp` / `inv` / `d_inv_dz` / `d_warp_dy` per column; batch helpers for `Array2` y and yvar.
- [ ] Thread `y_bounds: Array2<f64>` (always present; default `[[-∞,∞]; m]`) through `EpistemicNearestNeighbors` construction and `add`.
- [ ] Ingress: warp y / yvar before `append_rows` and before moment updates for `y_scale`.
- [ ] Egress: inverse-warp in `posterior`, `batch_posterior`, `conditional_posterior`, `posterior_function_draw`; inverse-warp y (and yvar via `1/(φ')²`) in `train_rows_at` / `row_y` / public observation accessors used by Python `_train_y`.
- [ ] Persist / reload `y_bounds` in `metadata.json` for disk backends; reject reopen mismatch if caller passes conflicting bounds.
- [ ] PyO3 + `enn_class.py`: expose `y_bounds=` on construction; keep posterior / draw return values in natural units.

**Validation:** `cargo test -p ennbo y_bounds`; new Rust tests for identity round-trip, logit/log branches, OOB reject, Jacobian yvar; Python `tests/test_enn_core.py` (or new `tests/test_y_bounds.py`) asserting `posterior.mu` in `(a,b)`, draws in `(a,b)`, and `_train_y` equals natural inputs after `add`.

### Phase 2 — Optimizer / surrogate config wiring

- [ ] Add `y_bounds` to Rust + Python `ENNSurrogateConfig`; plumb via `ConfigOverrides` / `create_optimizer_enn` into `ENNSurrogate` model construction.
- [ ] On first observation batch, validate `y_bounds.nrows() == num_metrics` (or broadcast a single pair to all metrics only if shape `(1,2)` is explicitly allowed — pick **strict `(num_metrics, 2)`** and document).
- [ ] Leave acquisition / incumbent on warped storage (no extra changes once model stores warped).
- [ ] Ensure `RustOptimizer.tell` continues to pass natural y; warp happens inside Rust model/surrogate, not in the Python x-unit wrapper.

**Validation:** `cargo test -p ennbo`; Python `tests/test_turbo_invariance.py` still passes (unbounded default); new optimizer test: `turbo_enn_config` with `y_bounds=[[0,1]]`, `tell` natural y in `(0,1)`, `ask` runs, and any exposed `y_obs()` matches natural y.

### Phase 3 — `ENNNormal.sample` + regression tests

- [ ] Teach `ENNNormal` (or posterior construction) to bound-aware-sample as in Q3 when bounds are non-identity; identity path unchanged.
- [ ] Tests: OOB `tell` raises; unbounded path bit-identical to pre-change fixtures where feasible; multi-metric per-column bounds; disk reopen restores bounds and natural-unit posterior.
- [ ] Update any public docs / examples that discuss y units only if they already document the ENN constructor (no new markdown files beyond what the API docstrings cover).

**Validation:** `pytest tests/test_y_bounds.py tests/test_enn_core.py tests/test_turbo_invariance.py tests/parity -q`; `cargo test -p ennbo`; `cargo test -p enn-py` (or project’s usual PyO3 test command). Confirm `ENNNormal.sample` draws lie in `(a,b)` when bounds are set.
