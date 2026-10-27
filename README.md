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
