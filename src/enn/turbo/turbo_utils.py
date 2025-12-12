from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    import numpy as np
    import torch
    from gpytorch.likelihoods import GaussianLikelihood
    from numpy.random import Generator
    from scipy.stats._qmc import QMCEngine

    from .turbo_gp import TurboGP
    from .turbo_gp_noisy import TurboGPNoisy


from enn.enn.enn_util import standardize_y


def _next_power_of_2(n: int) -> int:
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


@contextlib.contextmanager
def torch_rng_context(generator: torch.Generator | Any) -> Iterator[None]:
    import torch

    old_state = torch.get_rng_state()
    try:
        torch.set_rng_state(generator.get_state())
        yield
    finally:
        torch.set_rng_state(old_state)


def fit_gp(
    x_obs_list: list[float] | list[list[float]],
    y_obs_list: list[float] | list[list[float]],
    num_dim: int,
    *,
    yvar_obs_list: list[float] | None = None,
    num_steps: int = 50,
) -> tuple[
    "TurboGP | TurboGPNoisy | None",
    "GaussianLikelihood | None",
    float,
    float,
]:
    import numpy as np
    import torch
    from gpytorch.constraints import Interval
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.mlls import ExactMarginalLogLikelihood

    from .turbo_gp import TurboGP
    from .turbo_gp_noisy import TurboGPNoisy

    x = np.asarray(x_obs_list, dtype=float)
    y = np.asarray(y_obs_list, dtype=float)
    n = x.shape[0]
    if yvar_obs_list is not None:
        if len(yvar_obs_list) != len(y_obs_list):
            raise ValueError(
                f"yvar_obs_list length {len(yvar_obs_list)} != y_obs_list length {len(y_obs_list)}"
            )
    if n == 0:
        return None, None, 0.0, 1.0
    if n == 1:
        gp_y_mean = float(y[0])
        gp_y_std = 1.0
        return None, None, gp_y_mean, gp_y_std
    gp_y_mean, gp_y_std = standardize_y(y)
    y_centered = y - gp_y_mean
    z = y_centered / gp_y_std
    train_x = torch.as_tensor(x, dtype=torch.float64)
    train_y = torch.as_tensor(z, dtype=torch.float64)
    lengthscale_constraint = Interval(0.005, 2.0)
    outputscale_constraint = Interval(0.05, 20.0)
    if yvar_obs_list is not None:
        y_var = np.asarray(yvar_obs_list, dtype=float)
        train_y_var = torch.as_tensor(y_var / (gp_y_std**2), dtype=torch.float64)
        model = TurboGPNoisy(
            train_x=train_x,
            train_y=train_y,
            train_y_var=train_y_var,
            lengthscale_constraint=lengthscale_constraint,
            outputscale_constraint=outputscale_constraint,
            ard_dims=num_dim,
        ).to(dtype=train_x.dtype)
        likelihood = model.likelihood
    else:
        noise_constraint = Interval(5e-4, 0.2)
        likelihood = GaussianLikelihood(noise_constraint=noise_constraint).to(
            dtype=train_y.dtype
        )
        model = TurboGP(
            train_x=train_x,
            train_y=train_y,
            likelihood=likelihood,
            lengthscale_constraint=lengthscale_constraint,
            outputscale_constraint=outputscale_constraint,
            ard_dims=num_dim,
        ).to(dtype=train_x.dtype)
        likelihood.noise = torch.tensor(0.005, dtype=train_y.dtype)
    model.covar_module.outputscale = torch.tensor(1.0, dtype=train_x.dtype)
    model.covar_module.base_kernel.lengthscale = torch.full(
        (num_dim,), 0.5, dtype=train_x.dtype
    )
    model.train()
    likelihood.train()
    mll = ExactMarginalLogLikelihood(likelihood, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    for _ in range(num_steps):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
    model.eval()
    likelihood.eval()
    return model, likelihood, gp_y_mean, gp_y_std


def latin_hypercube(
    num_points: int, num_dim: int, *, rng: Generator | Any
) -> np.ndarray:
    import numpy as np

    x = np.zeros((num_points, num_dim))
    centers = (1.0 + 2.0 * np.arange(0.0, num_points)) / float(2 * num_points)
    for j in range(num_dim):
        x[:, j] = centers[rng.permutation(num_points)]
    pert = rng.uniform(-1.0, 1.0, size=(num_points, num_dim)) / float(2 * num_points)
    x += pert
    return x


def argmax_random_tie(values: np.ndarray | Any, *, rng: Generator | Any) -> int:
    import numpy as np

    if values.ndim != 1:
        raise ValueError(values.shape)
    max_val = float(np.max(values))
    idx = np.nonzero(values >= max_val)[0]
    if idx.size == 0:
        return int(rng.integers(values.size))
    if idx.size == 1:
        return int(idx[0])
    j = int(rng.integers(idx.size))
    return int(idx[j])


def sobol_perturb_np(
    x_center: np.ndarray | Any,
    lb: np.ndarray | list[float] | Any,
    ub: np.ndarray | list[float] | Any,
    num_candidates: int,
    mask: np.ndarray | Any,
    *,
    sobol_engine: QMCEngine | Any,
) -> np.ndarray:
    import numpy as np

    n_sobol = _next_power_of_2(num_candidates)
    sobol_samples = sobol_engine.random(n_sobol)[:num_candidates]
    lb_array = np.asarray(lb)
    ub_array = np.asarray(ub)
    pert = lb_array + (ub_array - lb_array) * sobol_samples
    candidates = np.tile(x_center, (num_candidates, 1))
    if np.any(mask):
        candidates[mask] = pert[mask]
    return candidates


def raasp(
    x_center: np.ndarray | Any,
    lb: np.ndarray | list[float] | Any,
    ub: np.ndarray | list[float] | Any,
    num_candidates: int,
    *,
    num_pert: int = 20,
    rng: Generator | Any,
    sobol_engine: QMCEngine | Any,
) -> np.ndarray:
    import numpy as np

    num_dim = x_center.shape[-1]
    prob_perturb = min(num_pert / num_dim, 1.0)
    mask = rng.random((num_candidates, num_dim)) <= prob_perturb
    ind = np.nonzero(~mask.any(axis=1))[0]
    if len(ind) > 0:
        mask[ind, rng.integers(0, num_dim, size=len(ind))] = True
    return sobol_perturb_np(
        x_center, lb, ub, num_candidates, mask, sobol_engine=sobol_engine
    )


def to_unit(x: np.ndarray | Any, bounds: np.ndarray | Any) -> np.ndarray:
    import numpy as np

    lb = bounds[:, 0]
    ub = bounds[:, 1]
    if np.any(ub <= lb):
        raise ValueError(bounds)
    return (x - lb) / (ub - lb)


def from_unit(x_unit: np.ndarray | Any, bounds: np.ndarray | Any) -> np.ndarray:
    import numpy as np

    lb = np.asarray(bounds[:, 0])
    ub = np.asarray(bounds[:, 1])
    return lb + x_unit * (ub - lb)


def gp_thompson_sample(
    model: Any,
    x_cand: np.ndarray | Any,
    num_arms: int,
    rng: Generator | Any,
    gp_y_mean: float,
    gp_y_std: float,
) -> np.ndarray:
    import gpytorch
    import numpy as np
    import torch

    x_torch = torch.as_tensor(x_cand, dtype=torch.float64)
    seed = int(rng.integers(2**31 - 1))
    gen = torch.Generator(device=x_torch.device)
    gen.manual_seed(seed)
    with (
        torch.no_grad(),
        gpytorch.settings.fast_pred_var(),
        torch_rng_context(gen),
    ):
        posterior = model.posterior(x_torch)
        samples = posterior.sample(sample_shape=torch.Size([1]))
    ts = samples[0].reshape(-1)
    scores = ts.detach().cpu().numpy().reshape(-1)
    scores = gp_y_mean + gp_y_std * scores
    shuffled_indices = rng.permutation(len(scores))
    shuffled_scores = scores[shuffled_indices]
    top_k_in_shuffled = np.argpartition(-shuffled_scores, num_arms - 1)[:num_arms]
    idx = shuffled_indices[top_k_in_shuffled]
    return idx
