"""Python facade and the Rust extension agree on the public optimizer and posterior."""

from __future__ import annotations

import numpy as np

from enn import EpistemicNearestNeighbors, _rust
from enn.enn.enn_fitter import ENNStatefulFitter
from enn.turbo.config.acq_type import AcqType
from enn.turbo.config.factory import lhd_only_config, turbo_enn_config, turbo_zero_config
from enn.turbo.config.morbo_tr_config import MorboTRConfig
from enn.turbo.config.multi_objective_config import MultiObjectiveConfig
from enn.turbo.rust_optimizer import create_optimizer
from enn.turbo.rust_optimizer_helpers import _config_to_rust_overrides


class _Seed:
    def __init__(self, seed: int) -> None:
        self.seed = seed

    def integers(self, *_args, **_kwargs) -> int:
        return self.seed


def _bounds(dim: int) -> np.ndarray:
    lo = np.linspace(-2.0, 0.0, dim)
    hi = lo + 3.0
    return np.stack([lo, hi], axis=1)


def _assert_in_bounds(x: np.ndarray, bounds: np.ndarray) -> None:
    assert x.shape[1] == bounds.shape[0]
    assert np.all(x >= bounds[:, 0])
    assert np.all(x <= bounds[:, 1])


def _pair(bounds, config, kind: str, seed: int):
    py = create_optimizer(bounds=bounds, config=config, rng=_Seed(seed))
    overrides = _config_to_rust_overrides(config)
    num_init = config.init.num_init
    if kind == "enn":
        k = None if config.surrogate.k is None else int(config.surrogate.k)
        rust = _rust.create_optimizer_enn(
            bounds, k, num_init, seed, config_overrides=overrides
        )
    elif kind == "zero":
        rust = _rust.create_optimizer_zero(
            bounds, num_init, seed, config_overrides=overrides
        )
    else:
        rust = _rust.create_optimizer_lhd(
            bounds, num_init, seed, config_overrides=overrides
        )
    x_py = py.ask(2)
    x_rust = np.asarray(rust.ask(2), dtype=float)
    assert np.allclose(x_py, x_rust)
    _assert_in_bounds(x_py, bounds)
    y = np.zeros((2, 1 if kind != "morbo" else 2))
    py.tell(x_py, y)
    rust.tell(x_rust, y, None)
    assert np.allclose(py._x_obs, np.asarray(rust.x_obs()))


def test_turbo_enn_natural_ask_matches_rust():
    for dim in (2, 20, 60):
        bounds = _bounds(dim)
        cfg = turbo_enn_config(num_init=2, acq_type=AcqType.PARETO)
        _pair(bounds, cfg, "enn", seed=11 + dim)


def test_turbo_zero_and_lhd_match_rust():
    bounds = _bounds(2)
    _pair(bounds, turbo_zero_config(num_init=2), "zero", seed=3)
    _pair(bounds, lhd_only_config(num_init=2), "lhd", seed=4)


def test_morbo_two_metrics_natural_ask():
    bounds = _bounds(2)
    cfg = turbo_enn_config(
        num_init=2,
        acq_type=AcqType.PARETO,
        trust_region=MorboTRConfig(multi_objective=MultiObjectiveConfig(num_metrics=2)),
    )
    py = create_optimizer(bounds=bounds, config=cfg, rng=_Seed(9))
    x = py.ask(2)
    _assert_in_bounds(x, bounds)
    y = np.array([[0.1, 0.4], [0.2, 0.3]])
    py.tell(x, y)
    assert py._y_obs.shape == (2, 2)


def test_posterior_mu_se_and_fitted_params():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(24, 2))
    y = x[:, :1].copy()
    model = EpistemicNearestNeighbors(x, y)
    from enn.enn.enn_params import ENNParams

    params = ENNParams(k_num_neighbors=4, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.1)
    post = model.posterior(x[:4], params=params)
    assert post.mu.shape == (4, 1)
    assert post.se.shape == (4, 1)
    assert np.all(np.isfinite(post.mu))
    assert np.all(post.se >= 0)
    fitter = ENNStatefulFitter(k=4, rng=np.random.default_rng(1))
    fitter.tell(x, y)
    params = fitter.ask(model, num_fit_candidates=6, num_fit_samples=4)
    assert params.k_num_neighbors > 0
    assert np.isfinite(params.epistemic_variance_scale)
