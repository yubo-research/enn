# Optimization Algorithms

This document describes the core optimization strategies available in the `ennbo` package.

## TURBO_ONE (Standard TuRBO)
The default high-performance strategy for single-objective optimization using a Gaussian Process (GP) surrogate.
- **Surrogate**: Gaussian Process with ARD (Automatic Relevance Determination) kernels.
- **Trust Region**: Maintains a single local trust region that expands on success and contracts on failure.
- **Acquisition**: Uses Thompson Sampling select arms from candidates within the trust region.

## TURBO_ENN (Epistemic Nearest Neighbors TuRBO)
An scalable strategy designed for scalability, replacing the GP with an Epistemic Nearest Neighbors (ENN) model.
- **Surrogate**: Epistemic Nearest Neighbors (ENN). Provides fast, $O(N)$ updates and predictions.
- **Trust Region**: Similar to TuRBO-ONE, but uses ENN-specific incumbent selection.
- **Incumbent Selection**: Uses a "Top-K" denoising algorithm. It raw-filters the top $K$ observations and then selects the best among them. If `noise_aware` is enabled, it uses the ENN posterior mean for the final selection to ensure noise robustness; otherwise, it uses the raw observations. O(N) complexity is maintained.

## TURBO_ZERO (No-Surrogate TuRBO)
A baseline TuRBO implementation that operates without a surrogate model.
- **Surrogate**: None (`NoSurrogate`).
- **Trust Region**: Maintains the trust region logic (centering on the best raw observation).
- **Acquisition**: Randomly select a RAASP candidate in the trust region.

## LHD_ONLY (Baseline)
A non-iterative baseline strategy that uses space-filling designs.
- **Surrogate**: None.
- **Strategy**: Generates all points using a Latin Hypercube Design (LHD).

## Candidate Generation (RAASP)
Random Axis-Aligned Subspace Perturbation (RAASP) is the primary method for generating candidates within a trust region.
- **Perturbation**: Instead of perturbing all dimensions, RAASP only perturbs a subset of dimensions around the trust region center.
- **Probability**: The probability of perturbing a dimension is $\min(20/D, 1.0)$, where $D$ is the number of dimensions. At least one dimension is always perturbed.
- **Sampling**: Perturbed dimensions are filled using a Sobol sequence or uniform sampling within the trust region bounds, while non-perturbed dimensions remain at the center value.

## Approximate Nearest Neighbors (HNSW)
Hierarchical Navigable Small World (HNSW) is an optional driver for nearest neighbor search in ENN models.
- **Mechanism**: Builds a hierarchical graph structure for $O(\log N)$ search and incremental updates.
- **Scaling**: Essential for scaling to large $N$ and large $D$ where exhaustive search becomes a bottleneck.
- **Performance**: While it scales better asymptotically, it is slower than `IndexFlat` in the BO loops we've tried, up to N=100k.

## Anisotropy & Volume Normalization
Trust regions can be anisotropic, scaling differently in each dimension based on the surrogate's learned lengthscales.
- **Anisotropy**: GP lengthscales from the ARD kernel are used to scale the trust region half-widths. Dimensions with larger lengthscales (lower sensitivity) result in wider trust regions.
- **Volume-Based Normalization**: Lengthscales are normalized such that their product is 1 ($\prod \ell_i = 1$). This ensures that the trust region's volume depends only on the scalar `length` parameter, not on the relative sensitivities of the dimensions.
- **ENN/Baseline**: ENN and No-Surrogate modes use isotropic trust regions (all dimensions scaled equally).

## Acquisition Functions
Acquisition functions select the best points to sample from the generated candidates.

### Thompson Sampling (TS)
The default acquisition for `TURBO_ONE`.
- **Mechanism**: Samples $N$ function draws from the surrogate posterior and selects the candidate that maximizes each draw.
- **Multi-Objective**: Samples are scalarized using Chebyshev weights before selection.

### UCB (Upper Confidence Bound)
A classic acquisition balancing mean and uncertainty.
- **Formula**: $UCB = \mu + \beta \sigma$, where $\beta$ (default 1.0) controls the exploration-exploitation trade-off.
- **Multi-Objective**: UCB is computed for all metrics and then scalarized.

### Pareto Acquisition
A multi-objective-inspired acquisition that balances exploitation ($\mu$) and exploration ($\sigma$).
- **Mechanism**: Performs 2D non-dominated sorting on the $(\mu, \sigma)$ pairs of all candidates.
- **Selection**: Iteratively extracts Pareto fronts (maximizing both $\mu$ and $\sigma$) until the required number of arms is selected.

### HNR (Hit-and-Run) Optimizer
An optional refinement step for acquisition.
- **Mechanism**: Starts from the candidates selected by TS or UCB and performs a local gradient-free search (Hit-and-Run) to further improve the acquisition score.

## Multi-Objective Support (MORBO)
The trust region and incumbent selection logic can be extended to support multiple metrics ($m > 1$) using the MORBO (Multi-Objective Bayesian Optimization) framework.
- **Scalarization**: Uses a Chebyshev scalarization with random weights to convert multiple objectives into a single score for trust region length updates.
- **Incumbent Selection**:
    - **TR Center**: The trust region is centered on a point selected from the current Pareto front.
    - **TR Update**: Success is determined by improvement in the scalarized score.
- **Weights**: Scalarization weights are resampled on trust region restarts or on every proposal (depending on the `Rescalarize` policy).
- **Complexity**: Maintains efficiency by using $O(N \log^{m-1} N)$ Pareto front extraction (via `nds`) and $O(N)$ scalarization.
