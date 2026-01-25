from __future__ import annotations
import numpy as np
import pytest
from enn.turbo.components.acquisition import (
    HnRAcqOptimizer,
    ParetoAcqOptimizer,
    RandomAcqOptimizer,
    ThompsonAcqOptimizer,
    UCBAcqOptimizer,
)
from enn.turbo.components.builder import build_acquisition_optimizer, build_surrogate
from enn.turbo.components.protocols import (
    PosteriorResult,
    SurrogateResult,
)
from enn.turbo.components.surrogates import ENNSurrogate, GPSurrogate, NoSurrogate
from enn.turbo.config import (
    ENNSurrogateConfig,
    NoTRConfig,
    turbo_enn_config,
    turbo_one_config,
    turbo_zero_config,
)
from enn.turbo.optimizer import Optimizer, create_optimizer


def _make_test_data(n: int = 4, d: int = 2):
    x = np.array([[0.2, 0.3], [0.5, 0.5], [0.7, 0.8], [0.1, 0.9]])[:n, :d]
    y = np.array([0.5, 0.7, 0.3, 0.6])[:n]
    return x, y


def _make_candidates(n: int = 4, d: int = 2):
    return np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])[:n, :d]


def _fit_gp_surrogate(rng):
    surrogate = GPSurrogate()
    x, y = _make_test_data()
    surrogate.fit(x, y, None, num_steps=10, rng=rng)
    return surrogate


def _fit_no_surrogate(rng, n: int = 2):
    surrogate = NoSurrogate()
    x, y = _make_test_data(n=n)
    surrogate.fit(x, y, None, num_steps=0, rng=rng)
    return surrogate


def test_surrogate_result():
    result = SurrogateResult(model="test_model", lengthscales=np.array([1.0, 2.0]))
    assert result.model == "test_model"
    assert np.allclose(result.lengthscales, [1.0, 2.0])


def test_surrogate_result_defaults():
    result = SurrogateResult(model=None)
    assert result.model is None
    assert result.lengthscales is None


def test_posterior_result():
    mu = np.array([[1.0], [2.0]])
    sigma = np.array([[0.1], [0.2]])
    result = PosteriorResult(mu=mu, sigma=sigma)
    assert np.allclose(result.mu, mu)
    assert np.allclose(result.sigma, sigma)


def test_posterior_result_no_sigma():
    mu = np.array([[1.0], [2.0]])
    result = PosteriorResult(mu=mu)
    assert np.allclose(result.mu, mu)
    assert result.sigma is None


def test_no_surrogate_fit():
    surrogate = NoSurrogate()
    x, y = _make_test_data(n=2)
    result = surrogate.fit(x, y, None, num_steps=0, rng=np.random.default_rng(0))
    assert result.model is None
    assert result.lengthscales is None


def test_no_surrogate_predict():
    rng = np.random.default_rng(0)
    surrogate = _fit_no_surrogate(rng)
    x, _ = _make_test_data(n=2)
    posterior = surrogate.predict(x)
    assert posterior.mu.shape == (2, 1)
    assert posterior.sigma is None


def test_no_surrogate_sample():
    rng = np.random.default_rng(42)
    surrogate = _fit_no_surrogate(rng)
    x, _ = _make_test_data(n=2)
    samples = surrogate.sample(x, 3, rng)
    assert samples.shape == (3, 2, 1)


def test_gp_surrogate_init():
    surrogate = GPSurrogate()
    assert surrogate._model is None


def test_gp_surrogate_fit():
    surrogate = GPSurrogate()
    x, y = _make_test_data()
    rng = np.random.default_rng(42)
    result = surrogate.fit(x, y, None, num_steps=10, rng=rng)
    assert result.model is not None
    assert result.lengthscales is not None
    assert result.lengthscales.shape == (2,)


def test_gp_surrogate_predict():
    rng = np.random.default_rng(42)
    surrogate = _fit_gp_surrogate(rng)
    x, _ = _make_test_data()
    posterior = surrogate.predict(x)
    assert posterior.mu.shape == (4, 1)
    assert posterior.sigma.shape == (4, 1)


def test_gp_surrogate_predict_not_fitted_raises():
    surrogate = GPSurrogate()
    x = np.array([[0.5, 0.5]])
    with pytest.raises(RuntimeError, match="fitted"):
        surrogate.predict(x)


def test_enn_surrogate_init():
    config = ENNSurrogateConfig(k=5)
    surrogate = ENNSurrogate(config)
    assert surrogate._config.k == 5


def test_enn_surrogate_fit():
    config = ENNSurrogateConfig(k=3)
    surrogate = ENNSurrogate(config)
    x, y = _make_test_data()
    rng = np.random.default_rng(42)
    result = surrogate.fit(x, y, None, num_steps=0, rng=rng)
    assert result.model is not None


def test_enn_surrogate_predict():
    config = ENNSurrogateConfig(k=3)
    surrogate = ENNSurrogate(config)
    x, y = _make_test_data()
    rng = np.random.default_rng(42)
    surrogate.fit(x, y, None, num_steps=0, rng=rng)
    posterior = surrogate.predict(x)
    assert posterior.mu.shape == (4, 1)


def test_no_surrogate_get_incumbent_candidate_indices():
    surrogate = NoSurrogate()
    y_obs = np.array([0.1, 0.3, -0.2], dtype=float)
    indices = surrogate.get_incumbent_candidate_indices(y_obs)
    assert np.array_equal(indices, np.array([0, 1, 2], dtype=int))


def test_gp_surrogate_get_incumbent_candidate_indices():
    surrogate = GPSurrogate()
    y_obs = np.array([0.2, -0.1, 0.4], dtype=float)
    indices = surrogate.get_incumbent_candidate_indices(y_obs)
    assert np.array_equal(indices, np.array([0, 1, 2], dtype=int))


def test_enn_surrogate_get_incumbent_candidate_indices_top_k():
    config = ENNSurrogateConfig(k=2)
    surrogate = ENNSurrogate(config)
    y_obs = np.array([0.1, 0.9, 0.3, 0.8], dtype=float)
    indices = surrogate.get_incumbent_candidate_indices(y_obs)
    assert np.array_equal(np.sort(indices), np.array([1, 3], dtype=int))


@pytest.mark.parametrize(
    "optimizer_cls,surrogate_fn",
    [
        (RandomAcqOptimizer, _fit_no_surrogate),
        (ThompsonAcqOptimizer, _fit_no_surrogate),
    ],
)
def test_acquisition_optimizer_select_with_no_surrogate(optimizer_cls, surrogate_fn):
    optimizer = optimizer_cls()
    x_cand = _make_candidates()
    rng = np.random.default_rng(42)
    surrogate = surrogate_fn(rng)
    selected = optimizer.select(x_cand, 2, surrogate, rng)
    assert selected.shape == (2, 2)


def test_ucb_acq_optimizer_init():
    optimizer = UCBAcqOptimizer(beta=2.0)
    assert optimizer._beta == 2.0


def test_ucb_acq_optimizer_select():
    optimizer = UCBAcqOptimizer(beta=1.0)
    rng = np.random.default_rng(42)
    surrogate = _fit_gp_surrogate(rng)
    x_cand = _make_candidates()
    selected = optimizer.select(x_cand, 2, surrogate, rng)
    assert selected.shape == (2, 2)


def test_pareto_acq_optimizer_select():
    optimizer = ParetoAcqOptimizer()
    rng = np.random.default_rng(42)
    config = ENNSurrogateConfig(k=3)
    surrogate = ENNSurrogate(config)
    x, y = _make_test_data()
    surrogate.fit(x, y, None, num_steps=0, rng=rng)
    x_cand = _make_candidates()
    selected = optimizer.select(x_cand, 2, surrogate, rng)
    assert selected.shape == (2, 2)


def test_hnr_acq_optimizer_init():
    base = UCBAcqOptimizer(beta=1.0)
    optimizer = HnRAcqOptimizer(base, num_iterations=10)
    assert optimizer._num_iterations == 10


@pytest.mark.parametrize(
    "base_cls,surrogate_fn",
    [
        (UCBAcqOptimizer, _fit_gp_surrogate),
        (ThompsonAcqOptimizer, _fit_no_surrogate),
    ],
)
def test_hnr_acq_optimizer_select(base_cls, surrogate_fn):
    if base_cls == UCBAcqOptimizer:
        base = base_cls(beta=1.0)
    else:
        base = base_cls()
    optimizer = HnRAcqOptimizer(base, num_iterations=5)
    rng = np.random.default_rng(42)
    surrogate = surrogate_fn(rng)
    x_cand = _make_candidates()
    selected = optimizer.select(x_cand, 2, surrogate, rng)
    assert selected.shape == (2, 2)


@pytest.mark.parametrize(
    "config_fn,expected_cls",
    [
        (turbo_one_config, GPSurrogate),
        (turbo_zero_config, NoSurrogate),
        (turbo_enn_config, ENNSurrogate),
    ],
)
def test_build_surrogate(config_fn, expected_cls):
    config = config_fn()
    surrogate = build_surrogate(config)
    assert isinstance(surrogate, expected_cls)


@pytest.mark.parametrize(
    "config_fn,expected_cls",
    [
        (turbo_zero_config, RandomAcqOptimizer),
        (turbo_one_config, ThompsonAcqOptimizer),
        (turbo_enn_config, ParetoAcqOptimizer),
    ],
)
def test_build_acquisition_optimizer(config_fn, expected_cls):
    config = config_fn()
    optimizer = build_acquisition_optimizer(config)
    assert isinstance(optimizer, expected_cls)


def test_surrogate_protocol_compliance():
    for cls in [NoSurrogate, GPSurrogate]:
        assert hasattr(cls, "fit")
        assert hasattr(cls, "predict")
        assert hasattr(cls, "sample")


def test_acquisition_optimizer_protocol_compliance():
    for cls in [
        RandomAcqOptimizer,
        ThompsonAcqOptimizer,
        UCBAcqOptimizer,
        ParetoAcqOptimizer,
    ]:
        assert hasattr(cls, "select")


def test_trust_region_protocol_from_config_build():
    tr_config = NoTRConfig()
    rng = np.random.default_rng(42)
    tr = tr_config.build(num_dim=3, rng=rng)
    assert hasattr(tr, "length")
    assert hasattr(tr, "compute_bounds_1d")
    assert hasattr(tr, "update")
    assert hasattr(tr, "needs_restart")
    assert hasattr(tr, "restart")


def test_optimizer_init():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    config = turbo_zero_config(num_init=4)
    rng = np.random.default_rng(42)
    surrogate = NoSurrogate()
    acq_optimizer = RandomAcqOptimizer()
    opt = Optimizer(
        bounds=bounds,
        config=config,
        rng=rng,
        surrogate=surrogate,
        acquisition_optimizer=acq_optimizer,
    )
    assert opt is not None
    assert opt._num_dim == 2
    init = opt.init_progress
    assert init is not None
    assert init == (0, 4)


def test_optimizer_init_via_factory():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    config = turbo_zero_config(num_init=4)
    rng = np.random.default_rng(42)
    opt = create_optimizer(bounds=bounds, config=config, rng=rng)
    assert opt is not None
    assert isinstance(opt, Optimizer)


def test_optimizer_fallback_during_init():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    config = turbo_one_config(num_init=10, num_candidates=16)
    rng = np.random.default_rng(42)
    opt = create_optimizer(bounds=bounds, config=config, rng=rng)
    x1 = opt.ask(num_arms=2)
    y1 = -np.sum(x1**2, axis=1)
    opt.tell(x1, y1)
    x2 = opt.ask(num_arms=2)
    assert x2.shape == (2, 2)
    init = opt.init_progress
    assert init is not None
    init_idx, num_init = init
    assert init_idx < num_init
