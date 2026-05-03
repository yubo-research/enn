# ennbo

Epistemic Nearest Neighbor Bayesian Optimization: a fast surrogate for Bayesian optimization.

ENN estimates a function's value and epistemic uncertainty using K-Nearest Neighbors. Queries take O(N ln K) time versus O(N²) for exact GPs.

## Features

- **EpistemicNearestNeighbors** — ENN surrogate with posterior computation
- **TuRBO-ENN optimizer** — Thompson sampling, UCB, RAASP candidate generation
- Neighbor search via Faiss (`IndexFlatL2`, `IndexHNSWFlat`-style factory string `HNSW32`)

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
ennbo = "0.1"
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
