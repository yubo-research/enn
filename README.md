# Epistemic Nearest Neighbors
A fast, alternative surrogate for Bayesian optimization

ENN estimates a function's value and associated epistemic uncertainty using a K-Nearest Neighbors model. Queries take $O(N \ln K)$ time, where $N$ is the number of observations available for KNN lookups, and measured running times stay small as $N$ grows. [1]

We also extend ENN with an alternative, disk-based approximate nearest neighbors backend based on B+ANN [4]. This reduces ENN and TuRBO-ENN to [effectively O(lnN)](https://github.com/yubo-research/enn/blob/main/reports/report_002/report_002.pdf) per iteration.

## Contents
- ENN surrogate, [`EpistemicNearestNeighbors`](https://github.com/yubo-research/enn/blob/main/src/enn/enn/enn.py) [1]
- TuRBO optimizer via [`create_optimizer`](https://github.com/yubo-research/enn/blob/main/src/enn/turbo/rust_optimizer.py) with config factories
	- `turbo_enn_config()` - TuRBO-ENN (Rust-backed by default)
	- `turbo_zero_config()` - TuRBO-zero (Rust-backed)
	- `lhd_only_config()` - LHD design on every `ask()` (Rust-backed)
The optimizer has an `ask()/tell()` interface. All `turbo_*()` methods follow TuRBO:
  - Generate candidates with RAASP [3] sampling.
  - Select a candidate with UCB or Thompson sampling (TuRBO-ENN), or randomly (TuRBO-zero).

- Overview of algorithms: [algos.pdf](docs/algos.pdf)



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


## Demonstration
[`demo_enn.ipynb`](https://github.com/yubo-research/enn/tree/main/examples/demo_enn.ipynb) - Shows how to use [`EpistemicNearestNeighbors`](https://github.com/yubo-research/enn/blob/main/src/enn/enn/enn.py) to build and query an ENN model.
[`demo_turbo_enn.ipynb`](https://github.com/yubo-research/enn/tree/main/examples/demo_turbo_enn.ipynb) - Shows how to use [`RustOptimizer`](https://github.com/yubo-research/enn/blob/main/src/enn/turbo/rust_optimizer.py) to optimize the Ackley function.



## Installation, MacOS

On my MacBook I can run into problems with dependencies and compatibilities.

On MacOS try:
```
micromamba env create -n ennbo -f admin/conda-macos.yml
micromamba activate ennbo
pip install --no-deps ennbo
pytest -sv tests
```

You may replace `micromamba` with `conda` and this will probably still work.

The commands above make sure
- faiss and SciPy come from one conda build, with a single OpenMP (`llvm-openmp` and `nomkl`) [faiss issue](https://github.com/faiss-wheels/faiss-wheels/issues/40).
- NumPy stays on a faiss-compatible build [faiss issue](https://github.com/faiss-wheels/faiss-wheels/issues/104).
- matplotlib's pin does not upgrade NumPy.
- `pip install --no-deps` leaves those pins in place.

Run tests with
```
pytest -x -sv tests
```
and they should all pass fairly quickly (~10s-30s).


If a duplicate OpenMP still crashes or hangs the process, try:
```
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
```
I don't recommend this, however, as it will slow things down.
