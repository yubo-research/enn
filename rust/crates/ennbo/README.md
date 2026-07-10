# ennbo

Epistemic Nearest Neighbor Bayesian Optimization: a fast surrogate for Bayesian optimization.

ENN estimates a function's value and epistemic uncertainty using K-Nearest Neighbors. Queries take O(N ln K) time versus O(N²) for exact GPs.

## Features

- **EpistemicNearestNeighbors** — ENN surrogate with posterior computation
- **TuRBO-ENN optimizer** — Thompson sampling, UCB, RAASP candidate generation
- Neighbor search via Faiss in-memory (`IndexDriver::Exact`)
- Disk mode: mmap `train_*.bin` + B+ANN index under `work_dir`
  - `IndexDriver::BpAnnDisk` — disk-backed B+ANN index via the `bpann` crate

### Disk layout (`bpann_disk`)

```
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
ENN_WORK_DIR=/tmp/enn_work cargo test -p ennbo disk_bpann
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
use ennbo::{EpistemicNearestNeighbors, ENNParams, index::IndexDriver};

let train_x = array![[0.0, 0.0], [1.0, 1.0]];
let train_y = array![[0.0], [1.0]];
let model = EpistemicNearestNeighbors::new(
    train_x,
    train_y,
    None,
    false,
    IndexDriver::Exact,
)?;

let params = ENNParams::new(5, 1.0, 0.1)?;
let out = model.posterior(&query_x.view(), &params, &Default::default())?;
```

## Python bindings

The [ennbo](https://pypi.org/project/ennbo/) Python package provides PyO3 bindings:

```bash
pip install ennbo[with-deps]
```

## License

MIT. See [repository](https://github.com/yubo-research/enn) for details.
