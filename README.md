# Epistemic Nearest Neighbors
A fast, alternative surrogate for Bayesian optimization

ENN estimates a function's value and associated epistemic uncertainty using k-nearest neighbors (KNN) or approximate nearest neighbors (ANN). Queries take O(Nlnk) time (KNN) or O(k lnN) time (ANN),
where N is the number of observations available for KNN lookups and k is the number of neighbors.

## Code
- Python API: [`src/enn/__init__.py`](https://github.com/yubo-research/enn/blob/main/src/enn/__init__.py)
- Rust API: [`rust/crates/ennbo/src/lib.rs`](https://github.com/yubo-research/enn/blob/main/rust/crates/ennbo/src/lib.rs)

## Contents
- ENN surrogate, [`EpistemicNearestNeighbors`](https://github.com/yubo-research/enn/blob/main/src/enn/enn/enn_class.py) [1]
- TuRBO optimizer via [`create_optimizer`](https://github.com/yubo-research/enn/blob/main/src/enn/turbo/rust_optimizer.py) with config factories
	- [`turbo_enn_config()`](https://github.com/yubo-research/enn/blob/main/src/enn/turbo/config/factory.py) - TuRBO-ENN (Rust-backed by default)
	- [`turbo_zero_config()`](https://github.com/yubo-research/enn/blob/main/src/enn/turbo/config/factory.py) - TuRBO-zero (Rust-backed)
	- [`lhd_only_config()`](https://github.com/yubo-research/enn/blob/main/src/enn/turbo/config/factory.py) - LHD design on every `ask()` (Rust-backed)
The optimizer has an `ask()/tell()` interface. All `turbo_*()` methods follow TuRBO:
  - Generate candidates with RAASP [3] sampling.
  - Select a candidate with UCB or Thompson sampling (TuRBO-ENN), or randomly (TuRBO-zero).



[1] **M. Bafna, Jadhav, S. a., & Sweet, D., (2025).** Taking the GP Out of the Loop. *arXiv preprint arXiv:2506.12818*.
   https://arxiv.org/abs/2506.12818
[2] **Eriksson, D., Pearce, M., Gardner, J. R., Turner, R., & Poloczek, M. (2020).** Scalable Global Optimization via Local Bayesian Optimization. *Advances in Neural Information Processing Systems, 32*.
   https://arxiv.org/abs/1910.01739
[3] **Rashidi, B., Johnstonbaugh, K., & Gao, C. (2024).** Cylindrical Thompson Sampling for High-Dimensional Bayesian Optimization. *Proceedings of The 27th International Conference on Artificial Intelligence and Statistics* (pp. 3502–3510). PMLR.
[4] [4] **Tekin, S. F., & Bordawekar, R. (2025).** B+ANN: A Fast Billion-Scale Disk-based Nearest-Neighbor Index. *arXiv preprint arXiv:2511.15557*.
    https://arxiv.org/abs/2511.15557
   https://proceedings.mlr.press/v238/rashidi24a.html


## Installation
`pip install ennbo[with-deps]`
or
`cargo add ennbo`


Run tests with
```
make test
```


## Backend and input features

Python uses the same Rust implementations as the native Rust API. Choose a backend
and one `weighting` flag:

| Backend | Storage | `weighting=False` | `weighting=True` |
| --- | --- | --- | --- |
| `FLAT` | Memory | Raw distances | Running per-dimension scaling |
| `BPANN_DISK` | Disk, with an explicit `work_dir` | Fixed distances | AUTO metric learning |

`enn_storage`, `scale_x`, and `metric_learning` are no longer constructor or optimizer
override parameters. `work_dir` is required for BPANN and rejected for FLAT.

```python
from enn import EpistemicNearestNeighbors
from enn.turbo.config import ENNIndexDriver, ENNSurrogateConfig

model = EpistemicNearestNeighbors(train_x, train_y, weighting=True)
disk_config = ENNSurrogateConfig(
    index_driver=ENNIndexDriver.BPANN_DISK,
    work_dir="./enn-work",
    weighting=True,
)
```

Both supervised models and `ENNSurrogateConfig` accept `input_transform` and
`semantic_only`. The default appends semantic dimensions to normalized parameter
dimensions; `semantic_only=True` uses just the projection. Weighting and tied
features can be enabled independently. `tied_dims` always indexes the resulting
feature columns. With FLAT weighting, tied columns keep scale 1; with AUTO, each
group shares a learned weight. Without weighting, the groups remain available as
metadata.

`CatEncoder` implements one-hot, Rademacher, and shared-hash representations in
Rust. Encode mixed inputs before fitting either a supervised model or an optimizer.
Supervised models read the encoded array's tie groups automatically; pass
`encoder.tied_dims()` in an optimizer's surrogate configuration. If numerical
optimizer proposals must correspond to categorical labels, use `encoder.snap`
with the allowed domains before evaluating and telling the resulting observations.
When composing categorical and semantic transforms, define semantic bounds and
names for the encoded columns. For semantic-only features, specify ties in the
projected columns.

The corresponding Rust APIs are `EnnLayout::new(driver, work_dir, weighting)`,
`EpistemicNearestNeighbors::new_configured`, `categorical_encoder::CatEncoder`,
and `semantic_projection::{projection, weighted, transform}`. Optimizer surrogate
configuration uses the same layout, tie validation, and semantic feature functions.

For semantic-only BPANN models, original parameter rows are stored alongside the
index in `original_inputs.bin`. Reopen with the same input transform and empty
training arrays to retain original-row access and fitting. Categorical encoders
preserve exact signed 64-bit ids when given integer arrays or mixed object arrays;
ids already rounded into floating-point inputs cannot be recovered.

`SemanticInputTransform` builds embeddings on the CPU with
[sentence-transformers-rs](https://github.com/jwnz/sentence-transformers-rs),
using `sentence-transformers/all-MiniLM-L6-v2` by default. Description file loading,
text deduplication, embedding, and projection construction run in Rust. Python
sentence-transformers and PyTorch are not required. The first construction fetches
model files into the Hugging Face cache; save the resulting transform with
`transform.save(path)` to load and use it later without loading an embedding model.
The `model=` argument accepts upstream's supported model presets (including
MiniLM, MPNet, LaBSE, BGE, and multilingual E5). Other models require an explicit
`encoder=` object implementing `encode(texts)`; unsupported presets raise a
`ValueError` before downloading files.

Use `SemanticInputTransform.from_input_output` for an additional prior based on
relationships between input parameters and output metrics:

```python
from enn import SemanticInputTransform, EpistemicNearestNeighbors

transform = SemanticInputTransform.from_input_output(
    names=["heater_power", "fan_speed"],
    bounds=[[0, 100], [0, 3000]],
    output_names=["temperature", "energy"],
    descriptions=["Power supplied to the heater", "Cooling fan rotation speed"],
    output_descriptions=["Chamber temperature", "Total electrical energy consumed"],
    relative_weight=0.25,
)
# train_x and train_y are your observed inputs and outputs.
model = EpistemicNearestNeighbors(train_x, train_y, input_transform=transform)
```

The transform builds signed cosine similarities between input and output text
embeddings, centers their columns, and retains the shortest SVD prefix reaching
`variance_cutoff` (default 0.9). It preserves cosine magnitudes before the SVD.
The retained factors have squared energy equal to their count before applying
`relative_weight`. Zero or constant similarities produce no factors. Original
input columns remain available with the default `semantic_only=False`.

This constructor needs no objective observations and uses one embedding call
with descriptions deduplicated across inputs and outputs. Description fallbacks,
local file URIs, `encoder=`, caching, and optimizer `input_transform=` work as
with the existing x-x constructor. `transform.save` and `SemanticInputTransform.load`
preserve the fixed projection without loading an encoder. All three priors use the same SVD variance cutoff. Output priors use an
uncentered SVD with orthonormal columns, capped at `n_outputs - 1` factors. The ordinary `SemanticInputTransform(...)` constructor continues
to build the existing x-x prior. Semantic similarity expresses a prior, rather
than evidence of causal influence; prediction improvements remain a hypothesis.

`AdaptiveSemanticENN` is an optional wrapper around Rust adaptation for learning from a stream of
complete numerical observations. It adapts a separate supervised input geometry
and signed output factors while preserving the supplied metadata definitions:

```python
from enn import AdaptiveSemanticENN

model = AdaptiveSemanticENN(input_transform=transform, capacity=300)
model.add(train_x, train_y)  # At least 12 rows in the initial batch; y is 2-D.
model.add(next_x, next_y)
posterior = model.posterior(test_x)
```

Pass `output_projection=` to adapt a `SemanticOutputProjection` as well. Each
`add` refits on a uniform replay sample of at most `capacity` rows, using separate
rows for fitting, candidate selection, and variance calibration. Candidates
include the fixed prior, supervised linear and quadratic-gradient geometries,
and independent, semantic, or learned output sharing. `model.choice` exposes the
selection. Learned input maps retain the metadata projection's factor count;
all priors initialize their count with the shared variance cutoff (default 0.9).
Semantic and learned output operators use `(1-alpha) I + alpha L Lᵀ`, with
`alpha` fitted on selection rows and exposed as `model.choice.alpha`. The low-rank
candidate remains the projector alone. Replay, splitting, fitting, selection, and
calibration run in Rust; Python wraps the public API.
Model read-back returns stored original input and output rows, including on disk.
A raw-input candidate makes zero semantic mixing available during selection. Both fixed and learned
input distances use a square matrix root, keeping the indexed input dimension
constant without discarding original coordinates. The public metadata transform
keeps its existing augmented-feature representation.
This is bounded replay with an epoch refit on every batch, suitable
for stationary streams; it does not preserve a full-history index or favor
recent observations under drift. It returns marginal means and standard errors,
without joint covariance or optimizer integration. Its optional linear mean is
disabled by default. Run `python ops/evaluate.py run short/adapt_xx` (or
`short/adapt_xy`, `short/adapt_yy`) for the controlled synthetic evaluations.
The larger comparison is reproducible with
`PYTHONPATH=src:. python scripts/evaluate_adapt.py`; see `report-adapt.pdf` for
results and limitations.

Adaptation shares each geometry's ENN index and transformed validation inputs
across neighbor counts. Mixed-count Flat batches use the scalar posterior fast
path. A small-workload certificate shares float64 neighbors when it proves
membership in every original float32 candidate set; ambiguous boundaries retain
separate searches. Small Flat searches run serially and restore the
caller's OpenMP thread budget afterward; larger searches retain parallelism.
By default, geometry weights 3 and 30 are tuned. Pass
`model.add(x, y, geometry_weight=30)` on each update to hold that mixing weight
fixed while retaining every geometry family, output operator, neighbor count,
and factor rank. This reduces selection work but can change the chosen model;
metadata embedding weights remain unchanged. Replay sampling, data splits, and
variance calibration retain the same protocol in both modes. Reproduce the runtime
and prediction comparison with
`PYTHONPATH=src:. python scripts/benchmark_adapt.py --output DIR`;
`--reference DIR` loads saved `adaptive.py` and `adaptive_model.py` files and,
when present, their saved native extension. Add `--geometry-weight 30` to measure fixed-weight updates.
See `report-adapt-speed.pdf` for measurements, the assessment
of recursive updates, and a bootstrap sensitivity analysis of learned geometry.

BPANN disk queries use the same tree search at every observation count. Fast
candidates receive float64 distance refinement with ten additional candidates at every N; a conservative rounding-error
bound requests float64 leaf scoring when the boundary is ambiguous. The bound
tracks metric rescaling and underflow, independently of N. The former 8,192-row
exhaustive-search transition is removed. AUTO continues fitting and
updating throughout the stream; its leave-one-out calculation shares neighbors
and grid weights across output columns. Approximate tree search retains its
logarithmically growing leaf budget. Accepted repartitions reuse centroids from
their acceptance calculation, avoiding duplicate weighted means. Repartitioning
receives one leaf of work credit per added row, spreading all 16 passes over
more additions. Refits, coordinate rescaling and I/O can still cause latency
spikes; the measured tradeoffs through one million rows are in
`report-smooth-adapt.pdf`.
