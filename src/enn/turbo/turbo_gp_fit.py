from __future__ import annotations
from typing import Any
from enn.enn.enn_util import standardize_y
from .types import GPDataPrep, GPFitResult


def _prepare_gp_data(
    x_obs_list: list, y_obs_list: list, yvar_obs_list: list | None
) -> GPDataPrep:
    import numpy as np
    import torch

    x = np.asarray(x_obs_list, dtype=float)
    y = np.asarray(y_obs_list, dtype=float)
    if y.ndim not in (1, 2):
        raise ValueError(y.shape)
    is_multi = y.ndim == 2 and y.shape[1] > 1
    if yvar_obs_list is not None:
        if len(yvar_obs_list) != len(y_obs_list):
            raise ValueError(
                f"yvar_obs_list length {len(yvar_obs_list)} != y_obs_list length {len(y_obs_list)}"
            )
        if is_multi:
            raise ValueError("yvar_obs_list not supported for multi-output GP")
    if is_multi:
        y_mean, y_std = y.mean(axis=0), y.std(axis=0)
        y_std = np.where(y_std < 1e-6, 1.0, y_std)
        z = (y - y_mean) / y_std
        train_y = torch.as_tensor(z.T, dtype=torch.float64)
    else:
        y_mean, y_std = standardize_y(y)
        z = (y - y_mean) / y_std
        train_y = torch.as_tensor(z, dtype=torch.float64)
    return GPDataPrep(
        train_x=torch.as_tensor(x, dtype=torch.float64),
        train_y=train_y,
        is_multi=is_multi,
        y_mean=y_mean,
        y_std=y_std,
        y_raw=y,
    )


def _build_gp_model(
    train_x: Any,
    train_y: Any,
    is_multi: bool,
    num_dim: int,
    *,
    yvar_obs_list: list | None,
    gp_y_std: Any,
    y: Any,
) -> tuple[Any, Any]:
    import numpy as np
    import torch
    from gpytorch.constraints import Interval
    from gpytorch.likelihoods import GaussianLikelihood
    from .turbo_gp import TurboGP
    from .turbo_gp_noisy import TurboGPNoisy

    ls_constr, os_constr = Interval(0.005, 2.0), Interval(0.05, 20.0)
    if yvar_obs_list is not None:
        y_var = np.asarray(yvar_obs_list, dtype=float)
        train_y_var = torch.as_tensor(y_var / (gp_y_std**2), dtype=torch.float64)
        model = TurboGPNoisy(
            train_x=train_x,
            train_y=train_y,
            train_y_var=train_y_var,
            lengthscale_constraint=ls_constr,
            outputscale_constraint=os_constr,
            ard_dims=num_dim,
        ).to(dtype=train_x.dtype)
        return model, model.likelihood
    noise_constr = Interval(5e-4, 0.2)
    num_out = int(y.shape[1]) if is_multi else None
    if is_multi:
        likelihood = GaussianLikelihood(
            noise_constraint=noise_constr, batch_shape=torch.Size([num_out])
        ).to(dtype=train_y.dtype)
    else:
        likelihood = GaussianLikelihood(noise_constraint=noise_constr).to(
            dtype=train_y.dtype
        )
    model = TurboGP(
        train_x=train_x,
        train_y=train_y,
        likelihood=likelihood,
        lengthscale_constraint=ls_constr,
        outputscale_constraint=os_constr,
        ard_dims=num_dim,
    ).to(dtype=train_x.dtype)
    likelihood.noise = (
        torch.full((num_out,), 0.005, dtype=train_y.dtype)
        if is_multi
        else torch.tensor(0.005, dtype=train_y.dtype)
    )
    return model, likelihood


def _init_gp_hyperparams(
    model: Any, is_multi: bool, num_dim: int, num_out: int | None, dtype: Any
) -> None:
    import torch

    if is_multi:
        model.covar_module.outputscale = torch.ones(num_out, dtype=dtype)
        model.covar_module.base_kernel.lengthscale = torch.full(
            (num_out, 1, num_dim), 0.5, dtype=dtype
        )
    else:
        model.covar_module.outputscale = torch.tensor(1.0, dtype=dtype)
        model.covar_module.base_kernel.lengthscale = torch.full(
            (num_dim,), 0.5, dtype=dtype
        )


def _train_gp(
    model: Any, likelihood: Any, train_x: Any, train_y: Any, num_steps: int
) -> None:
    import torch
    from gpytorch.mlls import ExactMarginalLogLikelihood

    model.train()
    likelihood.train()
    mll = ExactMarginalLogLikelihood(likelihood, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    for _ in range(num_steps):
        optimizer.zero_grad()
        loss = -mll(model(train_x), train_y)
        (loss.sum() if loss.ndim != 0 else loss).backward()
        optimizer.step()
    model.eval()
    likelihood.eval()


def fit_gp(
    x_obs_list: list[float] | list[list[float]],
    y_obs_list: list[float] | list[list[float]],
    num_dim: int,
    *,
    yvar_obs_list: list[float] | None = None,
    num_steps: int = 50,
) -> GPFitResult:
    import numpy as np

    x = np.asarray(x_obs_list, dtype=float)
    y = np.asarray(y_obs_list, dtype=float)
    n, is_multi = x.shape[0], y.ndim == 2 and y.shape[1] > 1
    if n == 0:
        return (
            GPFitResult(
                model=None,
                likelihood=None,
                y_mean=np.zeros(y.shape[1]),
                y_std=np.ones(y.shape[1]),
            )
            if is_multi
            else GPFitResult(model=None, likelihood=None, y_mean=0.0, y_std=1.0)
        )
    if n == 1 and is_multi:
        return GPFitResult(
            model=None,
            likelihood=None,
            y_mean=y[0].copy(),
            y_std=np.ones(int(y.shape[1]), dtype=float),
        )
    gp_data = _prepare_gp_data(x_obs_list, y_obs_list, yvar_obs_list)
    model, likelihood = _build_gp_model(
        gp_data.train_x,
        gp_data.train_y,
        gp_data.is_multi,
        num_dim,
        yvar_obs_list=yvar_obs_list,
        gp_y_std=gp_data.y_std,
        y=gp_data.y_raw,
    )
    _init_gp_hyperparams(
        model,
        gp_data.is_multi,
        num_dim,
        int(gp_data.y_raw.shape[1]) if gp_data.is_multi else None,
        gp_data.train_x.dtype,
    )
    _train_gp(model, likelihood, gp_data.train_x, gp_data.train_y, num_steps)
    return GPFitResult(
        model=model, likelihood=likelihood, y_mean=gp_data.y_mean, y_std=gp_data.y_std
    )
