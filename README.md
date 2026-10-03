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


