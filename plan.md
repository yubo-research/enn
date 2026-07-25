# Plan: Optional per-metric y bounds (output warp)

## User Request

Optional per-metric lower and/or upper bounds on y for the Rust and Python ENN / TuRBO-ENN APIs. Default `(-∞, +∞)` (identity). Users send natural-unit y and receive natural-unit posteriors (`mu`, `se`, draws, observation reads). The library warps internally and transparently.

## Current State

- **x bounds exist; y bounds do not.** Optimizer maps natural ↔ unit x in Python; Rust TR/acq stay in the unit box. No y transform today.
- **Storage:** `tell` / `add` store raw y; disk `train_y.bin` + `metadata.json` (`num_dim`, `num_metrics`, `scale_x`, …) — no `y_bounds`.
- **Scaling:** ENN does not center y. Per-metric `y_scale` = empirical std of stored y (`scale_from_moments`). GP `standardize_y` is out of scope.
- **Posterior / acq / incumbent:** All in stored-y units today (`posterior`, UCB, Thompson, incumbent, Morbo).
- **Config:** No `y_bounds` on `ENNSurrogateConfig`, `EpistemicNearestNeighbors`, or `ConfigOverrides` (Rust or Python).

## Design

### Warp

Strictly increasing φ so argmax(y) = argmax(z). UCB uses warped `μ_z + β·se_z` (not naturalized delta-method SE).

| Bounds | `z = φ(y)` | Inverse |
|---|---|---|
| `(-∞, +∞)` | `y` | identity |
| `(a, +∞)` | `log(y - a)` | `a + exp(z)` |
| `(-∞, b)` | `-log(b - y)` | `b - exp(-z)` |
| `(a, b)` | `logit((y-a)/(b-a))` | `a + (b-a)·σ(z)` |

Module: `rust/crates/ennbo/src/y_bounds.rs`. In-memory open sides = `±∞`; per-column pairs.

### Space split

Storage, fit, acquisition, and incumbent stay in warped `z`. Naturalize **only** at public API boundaries.

| Path | Space |
|---|---|
| Backend storage / model moments / `y_scale` | warped |
| Core / crate-internal posterior & row gather (used by surrogate, fitter, incumbent, Morbo) | warped |
| `ENNSurrogate::{predict,sample}`, UCB, Thompson | warped |
| Public Rust + PyO3 + Python: `posterior`, draws, `train_rows_at` / `row_y`, `Optimizer::y_obs` | natural |

**Dual API (required):** Public `EpistemicNearestNeighbors` / `Optimizer` methods return natural units. Surrogate, fitter, and incumbent call crate-internal warped entry points (e.g. `posterior_warped`, `train_rows_at_warped`, or `pub(crate)` row access on storage)—never the public naturalized wrappers.

**Internal `y_obs` call sites to switch to warped gather** (do not use naturalized public `y_obs()`):

- `Optimizer::update_incumbent` → `incumbent_tracker.rebuild`
- `morbo_sync_ranges_from_obs` in `strategy/mod.rs`
- Any other internal caller found by grep of `y_obs()` / `observations_y()` outside the public accessor itself

### Ingress / egress

- **Ingress:** Validate open interval; warp y and `yvar_z = (φ'(y))² · yvar_y`; reject non-finite `z`; then `append_rows` + moment updates. Warp a **copy** for `fitter.tell` / `incumbent_tracker.tell`—do not double-warp into storage.
- **Public egress:** `μ_nat = φ⁻¹(μ_z)`; `se_nat = |dφ⁻¹/dz|(μ_z) · se_z` (same factor on epi/ale); draws = `φ⁻¹` of warped samples.
- **`ENNNormal.sample`:** non-identity bounds → `φ⁻¹(φ(μ_nat) + (se_nat / |dφ⁻¹/dz|) · ε)`; identity → today’s `μ + se·ε`.

### API attachment

- Model: `y_bounds: Option<(num_metrics, 2)>`. None ⇒ all `(-∞,+∞)` for new models; disk reopen with None ⇒ load metadata.
- Optimizer: `ENNSurrogateConfig.y_bounds` → `construct_model` via `ConfigOverrides`. Shape **strict `(num_metrics, 2)`** (no `(1,2)` broadcast); validate when y width is known.
- GP / `turbo_one`: out of scope.
- `conditional_posterior`: warp `y_whatif` on ingress; public wrapper naturalizes like `posterior`.

### Validation & disk

- Open interval: `y <= a` or `y >= b` (finite sides) → error; require `a < b` when both finite; no silent clamp.
- ε policy: open membership **plus** finite warped `z` / `yvar_z` (near-bound overflow → error). No soft ε constant.
- `metadata.json`: array of `[lo, hi]` per metric; **finite number or `null`** for open side (strict JSON, not `Infinity`). Reopen: None → load; Some → must match (`null` ↔ `±∞`) or error.
- Affine-y invariance tests: unchanged on default unbounded path. Add separate bounded-y tests.

### Wiring notes

- `RustOptimizer.tell` still passes natural y; warp only inside Rust.
- Warp before `ENNFitter::tell` / `update_y`; fitter gathers via **warped** row APIs.
- Warp before `incumbent_tracker.tell` in `add_observations`; `obs_row_y` / Morbo / rebuild use warped gather.

## Phases

### Phase 1 — Warp primitives + model + public naturalize

- [ ] `y_bounds` module: validate; `warp` / `inv` / `d_inv_dz` / `d_warp_dy`; batch y/yvar; finite-`z` check.
- [ ] Thread `y_bounds` (default `[[-∞,∞]; m]`) through ENN construction / `add`; ingress warp before storage and moments.
- [ ] Keep core posterior APIs warped; add public naturalizing wrappers (Rust + PyO3 + Python) and `pub(crate)` warped gathers for fit/incumbent.
- [ ] Persist/reload `y_bounds` in metadata (`null` = open); reopen rules as above.
- [ ] Expose `y_bounds=` on Python/Rust construction.

**Validation:** `cargo test -p ennbo y_bounds`; round-trip / OOB / non-finite `z` / Jacobian yvar; Python `tests/test_y_bounds.py` — public `posterior.mu` and draws in `(a,b)`, `_train_y` equals natural inputs; stored warped row differs from natural for e.g. y=0.1 with bounds `(0,1)`.

### Phase 2 — Optimizer / surrogate

- [ ] `y_bounds` on Rust + Python `ENNSurrogateConfig` / `ConfigOverrides` / `create_optimizer_enn`.
- [ ] Validate `nrows == num_metrics` on first batch / construct with data.
- [ ] Warp before fitter tell; warp before `incumbent_tracker.tell`.
- [ ] Switch internal call sites listed above to warped gather; naturalize only public `y_obs()`.
- [ ] Keep `predict` / `sample` on warped core (UCB/Thompson in `z`).

**Validation:** unbounded invariance tests still pass; bounded optimizer test with open `(0,1)` — natural tell/ask, public `y_obs` natural, UCB/Thompson work.

### Phase 3 — `ENNNormal.sample` + regressions

- [ ] Bound-aware `ENNNormal.sample` (see egress); pass bounds/Jacobian from public posterior wrapper.
- [ ] Tests: OOB and non-finite `z`; multi-metric bounds; disk reopen None/conflict; unbounded bit-identical where feasible.
- [ ] Docstrings only (no new markdown files).

**Validation:** `pytest tests/test_y_bounds.py tests/test_enn_core.py tests/test_turbo_invariance.py tests/parity -q`; `cargo test -p ennbo`; `cargo test -p enn-py` (or usual PyO3 command).
