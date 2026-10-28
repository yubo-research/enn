# ennbo

Epistemic Nearest Neighbor Bayesian Optimization: a fast surrogate for Bayesian optimization.

ENN estimates a function's value and epistemic uncertainty using K-Nearest Neighbors. Queries take O(N ln K) time versus O(N²) for exact GPs.

## Features

- **EpistemicNearestNeighbors** — ENN surrogate with posterior computation
- **TuRBO-ENN optimizer** — Thompson sampling, UCB, RAASP candidate generation
- Neighbor search via Faiss in-memory (`IndexDriver::Flat`)
- Disk mode: mmap `train_*.bin` + B+ANN index under `work_dir`
  - `IndexDriver::BpAnnDisk` — disk-backed B+ANN index via the `ennbo-bpann` crate

### Disk layout (`bpann_disk`)

```text
work_dir/
  metadata.json       # index_backend: "bpann_disk", indexed_rows, num_dim, …
  train_x.bin         # f64 column mmap (posterior source of truth)
  train_y.bin
  train_yvar.bin      # optional
  index/
    header.json       # B+ANN shard metadata
    …                 # shard files managed by bpann
```

Incremental sync indexes rows `[indexed_rows..num_obs)` in chunks, updating `indexed_rows` in `metadata.json` after each chunk.

```bash
cargo test -p ennbo disk_bpann
```

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
ennbo = "0.2"
ndarray = "0.16"
```

```rust
use ndarray::array;
use ennbo::{EpistemicNearestNeighbors, ENNParams, IndexDriver, PosteriorFlags};

let train_x = array![[0.0, 0.0], [1.0, 1.0]];
let train_y = array![[0.0], [1.0]];
let model = EpistemicNearestNeighbors::new(
    train_x,
    train_y,
    None,
    false,
    IndexDriver::Flat,
).unwrap();
let query_x = array![[0.5, 0.5]];
let params = ENNParams::new(2, 1.0, 0.1).unwrap();
let out = model.posterior(&query_x.view(), &params, &PosteriorFlags::default()).unwrap();
assert_eq!(out.mu.shape(), &[1, 1]);
```

## Python bindings

The [ennbo](https://pypi.org/project/ennbo/) Python package provides PyO3 bindings:

```bash
pip install ennbo[with-deps]
```

## License

MIT. See [repository](https://github.com/yubo-research/enn) for details.

## Weighting and input features

Construct placement with `EnnLayout::new(driver, work_dir, weighting)`.
FLAT always stores in memory and uses data scaling when weighting is enabled.
BPANN_DISK requires an explicit directory and enables AUTO metric learning when
weighting is enabled. It never uses data scaling. `EnnLayout::flat(weighting)` and
`EnnLayout::disk(directory, weighting)` provide backend-specific constructors.

Pass the layout and optional feature-column tie groups to
`EpistemicNearestNeighbors::new_configured`. The constructor initializes AUTO and
ties in Rust, so direct callers and Python callers use the same policy.
`EnnOverrides::weighting` applies that policy to optimization.

`categorical_encoder::CatEncoder` plans, encodes, and snaps categorical blocks;
`semantic_projection` constructs projections and transforms bounded inputs.
`semantic_relationships::projection` constructs a shared input-output projection
from text embeddings and row indices (inputs followed by outputs). Each output
contributes one cosine-weighted input direction. Set the resulting projection,
scaled with `semantic_projection::weighted`, as `ENNSurrogateConfig::semantic_projection`
and keep `semantic_only` false to append features while preserving input variables.
Semantic features can augment or replace ordinary coordinates, with weighting
and ties enabled independently. Surrogate tie groups refer to feature coordinates.

Use `InputHistory::for_model` to retain original coordinates for a noninvertible
representation. It follows the model's layout: memory for FLAT, or the
`original_inputs.bin` sidecar for BPANN. Disk gathers read only requested rows,
and reopening checks the sidecar's dimensions, row count, and size against the model.

`CatEncoder::encode_i64` preserves exact integer category labels;
`encode_mixed` accepts `InputValue::Number` and `InputValue::Category` cells when
continuous and categorical columns require different numerical representations.
