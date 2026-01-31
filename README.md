# Epistemic Nearest Neighbors
A fast, alternative surrogate for Bayesian optimization

ENN estimates a function's value and associated epistemic uncertainty using a K-Nearest Neighbors model. Queries take $O(N lnK)$ time, where $N$ is the number of observations available for KNN lookups. Compare to an exact GP, which takes $O(N^2)$ time. Additionally, measured running times are very small compared to GPs and other alternative surrogates. [1]

## Contents
- ENN surrogate, [`EpistemicNearestNeighbors`](https://github.com/yubo-research/enn/blob/main/src/enn/enn/enn.py) [1]
- TuRBO-ENN optimizer via [`create_optimizer`](https://github.com/yubo-research/enn/blob/main/src/enn/turbo/optimizer.py) with config factories
	- `turbo_one_config()` - TuRBO [2], matching the reference implementation.
	- `turbo_enn_config()` - Uses ENN instead of GP.
	- `turbo_zero_config()` - No surrogate
	- `lhd_only_config()` - LHD design on every `ask()`. Good for a baseline and for testing.
The optimizer has an `ask()/tell()` interface. All `turbo_*()` methods follow TuRBO:
  - Generate candidates with RAASP [3] sampling.
  - Select a candidate with Thompson sampling (TuRBO-one), UCB (TuRBO-ENN), or randomly (TURBO-zero).


[1] **Sweet, D., & Jadhav, S. A. (2025).** Taking the GP Out of the Loop. *arXiv preprint arXiv:2506.12818*.
   https://arxiv.org/abs/2506.12818
[2] **Eriksson, D., Pearce, M., Gardner, J. R., Turner, R., & Poloczek, M. (2020).** Scalable Global Optimization via Local Bayesian Optimization. *Advances in Neural Information Processing Systems, 32*.
   https://arxiv.org/abs/1910.01739
[3] **Rashidi, B., Johnstonbaugh, K., & Gao, C. (2024).** Cylindrical Thompson Sampling for High-Dimensional Bayesian Optimization. *Proceedings of The 27th International Conference on Artificial Intelligence and Statistics* (pp. 3502–3510). PMLR.
   https://proceedings.mlr.press/v238/rashidi24a.html


## Installation
`pip install ennbo`

## Demonstration
[`demo_enn.ipynb`](https://github.com/yubo-research/enn/tree/main/examples/demo_enn.ipynb) - Shows how to use [`EpistemicNearestNeighbors`](https://github.com/yubo-research/enn/blob/main/src/enn/enn/enn.py) to build and query an ENN model.
[`demo_turbo_enn.ipynb`](https://github.com/yubo-research/enn/tree/main/examples/demo_turbo_enn.ipynb) - Shows how to use [`TurboOptimizer`](https://github.com/yubo-research/enn/blob/main/src/enn/turbo/turbo_optimizer.py) to optimize the Ackley function.



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
- You use the MacOS-specific PyTorch (with `mps`).
- You avoid having multiple, competing OpenMPs installed [PyTorch issue](https://github.com/pytorch/pytorch/issues/44282) [faiss issue](https://github.com/faiss-wheels/faiss-wheels/issues/40).
- You use old enough versions of NumPy and PyTorch to be compatible with faiss [faiss issue](https://github.com/faiss-wheels/faiss-wheels/issues/104).
- Prevent matplotlib's installation from upgrading your NumPy to an incompatible version.
- `ennbo`'s listed dependencies do not undo any of the above (which is fine b/c the above commands set the up correctly).

Run tests with
```
pytest -x -sv tests
```
and they should all pass fairly quickly (~10s-30s).


If your code still crashes or hangs, try this [hack](https://discuss.pytorch.org/t/ran-into-this-issue-while-executing/101460):
```
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
```
I don't recommend this, however, as it will slow things down.
