# Design 20260118


## Config hierarchy, semantically

Search space & sampling
  candidate_rv (SOBOL, UNIFORM)
  num_candidates

Initialization policy
  init_strategy, num_init

Surrogate model
  k
  ENN fitting
    num_fit_samples, num_fit_candidates
  scale_x

Acquisition / decision rule
  AcqType (THOMPSON, PARETO, UCB)

Trust region / locality
  TR length schedule
    length_init, length_min, length_max
  multi-objective shaping
    num_metrics, alpha
  rescalarize (ON_RESTART, ON_PROPOSE)

Observation handling
  trailing_obs


## Propose Config Hierarchy

Search space & sampling
  CandidateSamplingConfig
    candidate_rv
    num_candidates

Initialization policy
  InitPolicyConfig
    init_strategy
    num_init

Surrogate model
  SurrogateConfig
    k
    ENNFitConfig
      num_fit_samples
      num_fit_candidates
    scale_x

Acquisition / decision rule
  AcquisitionPolicyConfig
    AcqType

Trust region / locality
  TrustRegionConfig
    TRLengthConfig
      length_init
      length_min
      length_max
    MultiObjectiveConfig
      num_metrics
      alpha
    RescalePolicyConfig
      rescalarize

Observation handling
  ObservationHistoryConfig
    trailing_obs






---

# Design: Polymorphic Trust Region Architecture (No `isinstance` Dispatch)

## Problem

We currently have multiple “type switches” that dispatch on config/runtime types:

- `src/enn/turbo/impl_helpers.py:create_trust_region()` uses `if isinstance(tr_config, ...)` to decide which trust region implementation to construct.
- Other helpers branch on `config.is_morbo` and/or trust-region capabilities to decide how to interpret objective values (scalarize vs single objective).

This violates the project’s stated direction: **polymorphism over conditionals**. It also makes extensions painful (every new TR type requires touching central dispatch code).


## Goals

- **No `isinstance`/`if` ladders** to choose a trust region implementation.
- **Incumbent selection strategy** should be a configurable component (`IncumbentSelector`), not scattered conditionals in `Optimizer`/helpers.
- Keep the existing "unified TR interface" contract (update + candidate bounds), extended with `IncumbentSelector` composition.
- Preserve current behavior and tests; migration should be incremental.


## Proposed Architecture

### 1) Configs become polymorphic factories

Make `TrustRegionConfig` an ABC (or Protocol) with a method that constructs the runtime trust region.

Suggested API (shape only; names negotiable):

- `TrustRegionConfig.build(*, num_dim: int, num_arms: int, rng: Generator, num_metrics: int | None = None) -> TrustRegion`

Each config type implements `build()`:

- `NoTRConfig.build(...) -> NoTrustRegion(...)`
- `TurboTRConfig.build(...) -> TurboTrustRegion(config=self, ...)`
- `MorboTRConfig.build(...) -> MorboTrustRegion(config=self, ...)` (no `num_metrics` required)

Key point: **dispatch happens via virtual method call** (`cfg.trust_region.build(...)`), not a central `if isinstance(...)` switch.

This deletes `impl_helpers.create_trust_region()` entirely.


### 2) Runtime trust regions accept their config

Instead of exploding `TurboTrustRegion(...)` constructors into many scalar parameters, pass the config object:

- `TurboTrustRegion(*, config: TurboTRConfig, num_dim: int, num_arms: int)`
- `MorboTrustRegion(*, config: MorboTRConfig, num_dim: int, num_arms: int, rng: Generator)` (infers `num_metrics` on first `tell()`/`update()`)

Benefits:

- One place to interpret config values (defaults, invariants).
- Cleaner call sites; less argument drift when config evolves.


### 3) Incumbent selection is a separate component

Current pattern:

- Callers ask "is this morbo?" and then do different logic (scalarize vs pick max).
- Incumbent selection logic is scattered and depends on surrogate type (noise-aware vs noise-oblivious) and num_metrics.

Instead, introduce `IncumbentSelectorConfig` / `IncumbentSelector` as a dedicated component:

- `IncumbentSelectorConfig.build(...) -> IncumbentSelector`
- `IncumbentSelector.select(y_obs: np.ndarray, mu_obs: np.ndarray | None, rng: Generator) -> int`

The selector encapsulates the strategy for "which observation is best?":

- **Noise-oblivious, single-objective**: pick obs with highest `y` value
- **Noise-oblivious, multi-objective**: pick obs with highest `Chebyshev(y)` value
- **Noise-aware, single-objective**: pick obs with highest `mu(x_obs)` value
- **Noise-aware, multi-objective**: pick obs with highest `Chebyshev(mu)` value
- **ENN-specific**: top-K by observed `y`, then transform to `mu(x_obs)`, then pick highest

For multi-objective, Chebyshev weights are sampled either (a) once per TR reset, or (b) once per batch—configurable on the selector.

`TrustRegion` composes or receives an `IncumbentSelector` and delegates to it for center/restart decisions. This keeps `TrustRegion` focused on geometry (bounds, shrink/expand, restart detection).

`Acquisition` remains responsible for scoring *candidates* (UCB, Thompson, etc.)—no change there.

This removes `config.is_morbo` conditionals from non-TR code and makes incumbent selection strategies testable in isolation.


### 4) Optional: registry-based plugins (without `if` ladders)

If we want third-party TR plugins without importing their config classes in core:

- Keep `TrustRegionConfig.build()` as the primary path.
- Optionally add a registry as a fallback for “data-only” configs, but **avoid `if/elif` switches** by using dictionary lookup keyed by config type.

This keeps extension points while preserving the “no branching in callers” constraint.

### 5) Initialization is also a polymorphic component (single `Optimizer`)

We should not have multiple optimizer “modes” (or multiple optimizer classes). There should be **one `Optimizer`** with an initialization component that:

- generates initial points (e.g. LHD),
- tracks whether initialization is complete,
- and allows “LHD-only mode” to be expressed purely via configuration.

Concretely:

- Introduce an `InitStrategyConfig` ABC/Protocol with `build(...) -> InitStrategy`.
- `InitStrategy` is a runtime component with a minimal interface:
  - `ask(num_arms, *, bounds, rng, x_obs_count) -> np.ndarray | None`
  - `is_done(*, x_obs_count) -> bool`

Note: `InitStrategy` has no `tell()` method—initialization strategies ignore observations. The user still calls `Optimizer.tell()` during initialization (data is recorded), but init strategies don't use it.

The `Optimizer.ask()` flow becomes:

- ask init strategy for points; if it returns non-`None`, return those
- else proceed with TR + surrogate acquisition

“LHD-only mode” becomes a config that sets:

- `TrustRegionConfig = NoTRConfig()` (or a TR that never shrinks)
- `SurrogateConfig = NoSurrogateConfig()`
- `AcquisitionConfig = RandomAcquisitionConfig()`
- `InitStrategyConfig = LHDInitForeverConfig()` (explicit config class; no sentinel values)

So the optimizer never exits initialization, without adding any branching to `Optimizer`.


### 6) Component wiring pattern

`Optimizer` owns all runtime components and acts as the wiring hub. Each `build()` receives only what it strictly needs (not the whole Optimizer):

- `SurrogateConfig.build(num_dim, num_metrics)` — no dependencies
- `TrustRegionConfig.build(num_dim, num_arms, rng)` — no dependencies
- `IncumbentSelectorConfig.build(num_metrics)` — no dependencies
- `AcquisitionConfig.build()` — stateless
- `CandidateGeneratorConfig.build(num_dim)` — stateless
- `InitStrategyConfig.build(num_dim)` — no dependencies

Runtime methods receive collaborators at call time (not stored references):

- `Acquisition.score(candidates, mu, sigma, rng)` — receives surrogate predictions
- `CandidateGenerator.generate(bounds, center, rng, lengthscales)` — receives TR bounds and surrogate lengthscales
- `IncumbentSelector.select(y_obs, mu_obs, rng)` — receives observations and surrogate means

This avoids circular dependencies and keeps components testable in isolation.


### 7) Candidate generation details

`CandidateGenerator` generates candidates within TR bounds. The primary implementation is RAASP (Random Axis-Aligned Subspace Perturbation):

- RAASP *uses* Sobol or Uniform sampling as an internal parameter (not separate generator types)
- RAASP parameters: perturb probability `min(20/num_dim, 1.0)`, "at least one perturbed dim"
- RAASP needs surrogate lengthscales for anisotropic sampling → passed at call time: `generator.generate(bounds, center, rng, lengthscales=...)`

Config example: `RAASPCandidateConfig(base_sampler="sobol", perturb_prob=None)` where `None` means use default formula.


### 8) Observation window (trailing-obs)

When trailing-obs mode is enabled, old observations are truly discarded (not just ignored). This saves memory and prevents wasteful scans.

The optimizer owns the observation window and passes truncated data to all components:

- `Surrogate.fit(x, y, yvar)` receives only the trailing window
- `IncumbentSelector.select(...)` operates on the trailing window
- All components see a single, consistent set of observations

This is a configuration on the optimizer (e.g., `max_obs: int | None`), not per-component.


### 9) No-surrogate mode (`TURBO_ZERO`)

With `SurrogateConfig = NoSurrogateConfig()`:

- `NoSurrogate.predict(x)` returns the mean over observed `y` values (or zeros if no observations)
- Acquisition degrades to random selection with tie-breaking via `rng`
- This is handled naturally by the null-object pattern—no special branches needed


### 10) Trust region scope

For now, we use a single trust region (not multi-TR as in the original MORBO paper):

- `num_arms` refers to batch size, not number of TRs
- Single set of success/failure counters
- Restart logic stays inside `TrustRegion`; `InitStrategy` is *not* re-triggered on restart

Multi-TR support can be added later if needed, but is not in scope for this design.


## Migration Plan (Incremental)

1) **Introduce `TrustRegionConfig.build()`** on existing config classes (NoTR/TurboTR/MorboTR).
2) Update `Optimizer._create_trust_region()` to call `self._config.trust_region.build(...)`.
3) Delete `impl_helpers.create_trust_region()`. Update imports/callers accordingly.
4) **Introduce `IncumbentSelectorConfig` / `IncumbentSelector`** with implementations for:
   - Single-objective (noise-oblivious and noise-aware variants)
   - Multi-objective with Chebyshev scalarization
5) Refactor incumbent selection out of `Optimizer` and into `IncumbentSelector`:
   - `Optimizer._compute_scores()` → calls `incumbent_selector.select(...)`
   - `impl_helpers.get_x_center_fallback()` → delegated to TR + selector
6) Update `TrustRegion` to compose `IncumbentSelector` for center/restart decisions.
7) Ensure `pytest`, `ruff`, `kiss` remain green throughout.


## Guardrails (What to Ban)

- New `if isinstance(...)` / `elif isinstance(...)` switches for selecting implementations.
- Any “mode” flags (`cfg.is_morbo`, enums, booleans) in `Optimizer`/helpers to decide algorithm behavior.
- `None` as a component/config value (use explicit null-object configs/impls like `NoTRConfig`, `NoSurrogateConfig`, etc.).
- Sentinel values to change behavior (e.g. `np.inf`, magic strings, `-1`): represent variants via distinct config classes (e.g. `LHDInitConfig(num_init=int)` vs `LHDInitForeverConfig()`).
- Adding new TR types should require editing only:
  - the new `TrustRegionConfig` subclass
  - its runtime `TrustRegion` implementation
  - tests


## Coding Practices

### Liberal shape assertions

Use assertions liberally to verify array shapes at runtime, especially when arrays pass between components. This catches shape mismatches early (at the point of the bug) rather than later (as cryptic broadcasting errors).

Example pattern:

```python
samples = surrogate.sample(x_cand, num_arms, rng)

# Verify shape immediately after receiving from another component
assert samples.shape == (num_samples, num_candidates, num_metrics), (
    f"samples.shape={samples.shape}, "
    f"expected ({num_samples}, {num_candidates}, {num_metrics})"
)
```

Key places for shape assertions:
- Surrogate outputs (`predict`, `sample`)
- Acquisition optimizer inputs/outputs
- Trust region bounds
- Scalarization inputs/outputs

This is especially important when different implementations (e.g., GP vs ENN surrogates) must conform to the same shape contract.


## Notes / Non-goals

- This doc focuses on trust regions, but the same pattern applies elsewhere (surrogates, acquisition, etc.): configs should build runtime implementations, and runtime impls should own mode-specific behavior.



---



## Abstract component types (interfaces / ABCs)

- `TrustRegionConfig`: config-side factory (`build(...) -> TrustRegion`)
- `TrustRegion`: runtime TR geometry (update, bounds, shrink/expand, restart detection); composes `IncumbentSelector`
- `IncumbentSelectorConfig`: config-side factory (`build(...) -> IncumbentSelector`)
- `IncumbentSelector`: runtime incumbent selection (`select(y_obs, mu_obs, rng) -> int`); encapsulates noise-aware vs noise-oblivious, scalar vs Chebyshev strategies
- `SurrogateConfig`: config-side factory (`build(...) -> Surrogate`)
- `Surrogate`: runtime surrogate (fit/predict/sample; exposes `lengthscales`/posterior as needed)
- `AcquisitionConfig`: config-side factory (`build(...) -> Acquisition`)
- `Acquisition`: runtime acquisition scoring/draw policy (e.g. UCB, Thompson, Pareto)
- `CandidateGeneratorConfig`: config-side factory (`build(...) -> CandidateGenerator`)
- `CandidateGenerator`: runtime candidate generation (RAASP with Sobol/Uniform base; receives TR bounds and lengthscales at call time)
- `InitStrategyConfig`: config-side factory (`build(...) -> InitStrategy`)
- `InitStrategy`: runtime initialization policy (`ask(...)`, `is_done(...)`); no `tell()` method—init strategies ignore observations
