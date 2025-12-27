from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch
from gpytorch.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

from enn.turbo.turbo_gp_fit import fit_gp
from enn.turbo.turbo_gp_noisy import TurboGPNoisy


def _make_turbo_gp_noisy(
    *,
    train_x,
    train_y,
    train_y_var,
    ard_dims: int,
    learn_additional_noise: bool = False,
):
    lengthscale_constraint = Interval(0.005, 2.0)
    outputscale_constraint = Interval(0.05, 20.0)
    return TurboGPNoisy(
        train_x=train_x,
        train_y=train_y,
        train_y_var=train_y_var,
        lengthscale_constraint=lengthscale_constraint,
        outputscale_constraint=outputscale_constraint,
        ard_dims=ard_dims,
        learn_additional_noise=learn_additional_noise,
    )


def test_fit_gp_returns_model_with_valid_data():
    num_obs, num_dim = 20, 3
    x_obs = np.random.default_rng(0).random((num_obs, num_dim))
    y_obs = (
        x_obs.sum(axis=1) + 0.1 * np.random.default_rng(1).standard_normal(num_obs)
    ).tolist()
    model, likelihood, y_mean, y_std = fit_gp(
        x_obs.tolist(), y_obs, num_dim, num_steps=10
    )
    assert model is not None and likelihood is not None
    assert isinstance(y_mean, float) and isinstance(y_std, float) and y_std > 0.0


def test_fit_gp_returns_none_with_empty_data_and_returns_model_with_single_obs():
    num_dim = 2
    model_empty, likelihood_empty, mean_empty, std_empty = fit_gp(
        [], [], num_dim, num_steps=10
    )
    assert model_empty is None and likelihood_empty is None
    assert mean_empty == 0.0 and std_empty == 1.0

    x_single = np.random.default_rng(0).random((1, num_dim))
    model_single, likelihood_single, mean_single, std_single = fit_gp(
        x_single.tolist(), [1.0], num_dim, num_steps=0
    )
    assert model_single is not None and likelihood_single is not None
    assert mean_single == 1.0 and std_single == 1.0


def test_fit_gp_with_y_var_list_uses_noisy_model():
    num_obs, num_dim = 20, 3
    rng = np.random.default_rng(42)
    x_obs = rng.random((num_obs, num_dim))
    y_obs = (x_obs.sum(axis=1) + 0.1 * rng.standard_normal(num_obs)).tolist()
    y_var = rng.uniform(0.01, 0.1, size=num_obs).tolist()
    model, likelihood, y_mean, y_std = fit_gp(
        x_obs.tolist(), y_obs, num_dim, yvar_obs_list=y_var, num_steps=10
    )
    assert model is not None and isinstance(model, TurboGPNoisy)
    assert likelihood is not None and y_std > 0.0


def test_fit_gp_with_y_var_list_asserts_length():
    num_obs, num_dim = 10, 2
    rng = np.random.default_rng(0)
    x_obs = rng.random((num_obs, num_dim)).tolist()
    y_obs = rng.random(num_obs).tolist()
    y_var_wrong = rng.uniform(0.01, 0.1, size=num_obs - 2).tolist()
    with pytest.raises(ValueError):
        fit_gp(x_obs, y_obs, num_dim, yvar_obs_list=y_var_wrong, num_steps=5)


def test_turbo_gp_noisy_accepts_train_y_var():
    num_obs, num_dim = 10, 2
    rng = np.random.default_rng(42)
    train_x = torch.as_tensor(rng.random((num_obs, num_dim)), dtype=torch.float64)
    train_y = torch.as_tensor(
        train_x.sum(dim=1).numpy() + 0.1 * rng.standard_normal(num_obs),
        dtype=torch.float64,
    )
    train_y_var = torch.as_tensor(
        rng.uniform(0.01, 0.1, size=num_obs), dtype=torch.float64
    )
    model = _make_turbo_gp_noisy(
        train_x=train_x, train_y=train_y, train_y_var=train_y_var, ard_dims=num_dim
    )
    assert model is not None and isinstance(
        model.likelihood, FixedNoiseGaussianLikelihood
    )


def test_turbo_gp_noisy_forward_and_posterior():
    num_obs, num_dim = 15, 3
    rng = np.random.default_rng(123)
    train_x = torch.as_tensor(rng.random((num_obs, num_dim)), dtype=torch.float64)
    train_y = torch.as_tensor(
        train_x.sum(dim=1).numpy() + 0.05 * rng.standard_normal(num_obs),
        dtype=torch.float64,
    )
    train_y_var = torch.full((num_obs,), 0.01, dtype=torch.float64)
    model = _make_turbo_gp_noisy(
        train_x=train_x, train_y=train_y, train_y_var=train_y_var, ard_dims=num_dim
    )
    model.eval()
    model.likelihood.eval()
    test_x = torch.as_tensor(rng.random((5, num_dim)), dtype=torch.float64)
    with torch.no_grad():
        forward_output = model.forward(test_x)
        posterior_output = model.posterior(test_x)
    assert isinstance(forward_output, MultivariateNormal) and isinstance(
        posterior_output, MultivariateNormal
    )
    assert forward_output.mean.shape == (5,) and posterior_output.mean.shape == (5,)


def test_turbo_gp_noisy_trains_successfully():
    num_obs, num_dim = 20, 2
    rng = np.random.default_rng(999)
    train_x = torch.as_tensor(rng.random((num_obs, num_dim)), dtype=torch.float64)
    train_y = torch.as_tensor(
        train_x.sum(dim=1).numpy() + 0.1 * rng.standard_normal(num_obs),
        dtype=torch.float64,
    )
    train_y_var = torch.as_tensor(
        rng.uniform(0.005, 0.05, size=num_obs), dtype=torch.float64
    )
    model = _make_turbo_gp_noisy(
        train_x=train_x, train_y=train_y, train_y_var=train_y_var, ard_dims=num_dim
    )
    model.train()
    model.likelihood.train()
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    initial_loss = None
    for i in range(20):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        if i == 0:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()
    assert loss.item() <= initial_loss


def test_turbo_gp_noisy_with_zero_variance():
    num_obs, num_dim = 10, 2
    rng = np.random.default_rng(42)
    train_x = torch.as_tensor(rng.random((num_obs, num_dim)), dtype=torch.float64)
    train_y = torch.as_tensor(train_x.sum(dim=1).numpy(), dtype=torch.float64)
    train_y_var = torch.zeros(num_obs, dtype=torch.float64)
    from gpytorch.utils.warnings import NumericalWarning

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Very small noise values detected\..*",
            category=NumericalWarning,
        )
        model = _make_turbo_gp_noisy(
            train_x=train_x,
            train_y=train_y,
            train_y_var=train_y_var,
            ard_dims=num_dim,
            learn_additional_noise=True,
        )
    model.eval()
    model.likelihood.eval()
    test_x = torch.as_tensor(rng.random((3, num_dim)), dtype=torch.float64)
    with torch.no_grad():
        posterior = model.posterior(test_x)
    assert posterior.mean.shape == (3,)


def test_fit_gp_multi_output_can_trigger_non_scalar_backward_error():
    rng = np.random.default_rng(0)
    num_dim, num_metrics, n = 3, 2, 8
    x = rng.uniform(0.0, 1.0, size=(n, num_dim))
    y = rng.normal(size=(n, num_metrics))
    model, likelihood, _, _ = fit_gp(x.tolist(), y.tolist(), num_dim, num_steps=0)
    assert model is not None and likelihood is not None
    model.train()
    likelihood.train()
    train_x = model.train_inputs[0]
    train_y = model.train_targets
    output = model(train_x)
    mll = ExactMarginalLogLikelihood(likelihood, model)
    loss = -mll(output, train_y)
    assert tuple(loss.shape) == (num_metrics,)
    with pytest.raises(
        RuntimeError, match="grad can be implicitly created only for scalar outputs"
    ):
        loss.backward()


def test_fit_gp_multi_output_trains_without_scalar_backward_error():
    rng = np.random.default_rng(0)
    num_dim, num_metrics, n = 3, 2, 8
    x = rng.uniform(0.0, 1.0, size=(n, num_dim))
    y = rng.normal(size=(n, num_metrics))
    model, likelihood, gp_y_mean, gp_y_std = fit_gp(
        x.tolist(), y.tolist(), num_dim, num_steps=2
    )
    assert model is not None and likelihood is not None
    assert gp_y_mean.shape == (num_metrics,) and gp_y_std.shape == (num_metrics,)
