from __future__ import annotations
import warnings
import numpy as np
import pytest
import torch
from gpytorch.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from enn.turbo.components.gp_surrogate import GPSurrogate
from enn.turbo.turbo_gp import TurboGP
from enn.turbo.turbo_gp_base import TurboGPBase
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
    result = fit_gp(x_obs.tolist(), y_obs, num_dim, num_steps=10)
    assert result.model is not None and result.likelihood is not None
    assert (
        isinstance(result.y_mean, float)
        and isinstance(result.y_std, float)
        and result.y_std > 0.0
    )


def test_fit_gp_returns_none_with_empty_data_and_returns_model_with_single_obs():
    num_dim = 2
    result_empty = fit_gp([], [], num_dim, num_steps=10)
    assert result_empty.model is None and result_empty.likelihood is None
    assert result_empty.y_mean == 0.0 and result_empty.y_std == 1.0
    x_single = np.random.default_rng(0).random((1, num_dim))
    result_single = fit_gp(x_single.tolist(), [1.0], num_dim, num_steps=0)
    assert result_single.model is not None and result_single.likelihood is not None
    assert result_single.y_mean == 1.0 and result_single.y_std == 1.0


def test_fit_gp_with_y_var_list_uses_noisy_model():
    num_obs, num_dim = 20, 3
    rng = np.random.default_rng(42)
    x_obs = rng.random((num_obs, num_dim))
    y_obs = (x_obs.sum(axis=1) + 0.1 * rng.standard_normal(num_obs)).tolist()
    y_var = rng.uniform(0.01, 0.1, size=num_obs).tolist()
    result = fit_gp(x_obs.tolist(), y_obs, num_dim, yvar_obs_list=y_var, num_steps=10)
    assert result.model is not None and isinstance(result.model, TurboGPNoisy)
    assert result.likelihood is not None and result.y_std > 0.0


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
    result = fit_gp(x.tolist(), y.tolist(), num_dim, num_steps=0)
    assert result.model is not None and result.likelihood is not None
    result.model.train()
    result.likelihood.train()
    train_x = result.model.train_inputs[0]
    train_y = result.model.train_targets
    output = result.model(train_x)
    mll = ExactMarginalLogLikelihood(result.likelihood, result.model)
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
    result = fit_gp(x.tolist(), y.tolist(), num_dim, num_steps=2)
    assert result.model is not None and result.likelihood is not None
    assert result.y_mean.shape == (num_metrics,) and result.y_std.shape == (
        num_metrics,
    )


def test_gp_surrogate_predict_shapes_multi_output():
    rng = np.random.default_rng(0)
    num_dim, num_metrics, n = 3, 2, 6
    x = rng.uniform(0.0, 1.0, size=(n, num_dim))
    y = x @ rng.normal(size=(num_dim, num_metrics))
    surrogate = GPSurrogate()
    surrogate.fit(x, y, num_steps=2)
    posterior = surrogate.predict(x)
    mu = np.asarray(posterior.mu, dtype=float)
    sigma = np.asarray(posterior.sigma, dtype=float)
    assert mu.shape == (n, num_metrics)
    assert sigma.shape == (n, num_metrics)


def test_turbo_gp_init_and_forward():
    from gpytorch.likelihoods import GaussianLikelihood

    num_obs, num_dim = 10, 2
    rng = np.random.default_rng(42)
    train_x = torch.as_tensor(rng.random((num_obs, num_dim)), dtype=torch.float64)
    train_y = torch.as_tensor(
        train_x.sum(dim=1).numpy() + 0.1 * rng.standard_normal(num_obs),
        dtype=torch.float64,
    )
    lengthscale_constraint = Interval(0.005, 2.0)
    outputscale_constraint = Interval(0.05, 20.0)
    likelihood = GaussianLikelihood()
    model = TurboGP(
        train_x=train_x,
        train_y=train_y,
        likelihood=likelihood,
        lengthscale_constraint=lengthscale_constraint,
        outputscale_constraint=outputscale_constraint,
        ard_dims=num_dim,
    )
    assert model is not None
    assert isinstance(model, TurboGPBase)
    model.eval()
    likelihood.eval()
    test_x = torch.as_tensor(rng.random((5, num_dim)), dtype=torch.float64)
    with torch.no_grad():
        output = model.forward(test_x)
    assert isinstance(output, MultivariateNormal)
    assert output.mean.shape == (5,)


def test_turbo_gp_base_is_subclassed():
    assert TurboGPBase is not None
    assert issubclass(TurboGP, TurboGPBase)
