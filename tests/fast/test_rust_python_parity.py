"""Python facade and the Rust extension agree on the public optimizer and posterior."""

from __future__ import annotations

import numpy as np

from enn import EpistemicNearestNeighbors, _rust
from enn.enn.enn_fitter import ENNStatefulFitter
from enn.turbo.config.acq_type import AcqType
from enn.turbo.config.factory import (
    lhd_only_config,
    turbo_enn_config,
    turbo_zero_config,
)
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


def _rust_optimizer(bounds, config, kind: str, seed: int):
    overrides = _config_to_rust_overrides(config)
    num_init = config.init.num_init
    if kind in ("enn", "morbo"):
        k = None if config.surrogate.k is None else int(config.surrogate.k)
        return _rust.create_optimizer_enn(
            bounds, k, num_init, seed, config_overrides=overrides
        )
    if kind == "zero":
        return _rust.create_optimizer_zero(
            bounds, num_init, seed, config_overrides=overrides
        )
    return _rust.create_optimizer_lhd(
        bounds, num_init, seed, config_overrides=overrides
    )


def _pair(bounds, config, kind: str, seed: int, y: np.ndarray | None = None):
    py = create_optimizer(bounds=bounds, config=config, rng=_Seed(seed))
    rust = _rust_optimizer(bounds, config, kind, seed)
    x_py = py.ask(2)
    x_rust = np.asarray(rust.ask(2), dtype=float)
    assert np.allclose(x_py, x_rust)
    _assert_in_bounds(x_py, bounds)
    if y is None:
        y = np.zeros((2, 1))
    py.tell(x_py, y)
    rust.tell(x_rust, y, None)
    assert np.allclose(py._x_obs, np.asarray(rust.x_obs()))
    assert np.allclose(py._y_obs, np.asarray(rust.y_obs()))


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
    y = np.array([[0.1, 0.4], [0.2, 0.3]])
    _pair(bounds, cfg, "morbo", seed=9, y=y)


def test_posterior_mu_se_and_fitted_params():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(24, 2))
    y = x[:, :1].copy()
    query = x[:4]
    model = EpistemicNearestNeighbors(x, y)
    rust_model = _rust.EpistemicNearestNeighbors(x, y)
    from enn.enn.enn_params import ENNParams

    params = ENNParams(
        k_num_neighbors=4, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.1
    )
    post = model.posterior(query, params=params)
    mu, se, _se_epi, _se_ale, idx = rust_model.posterior(
        query,
        params.k_num_neighbors,
        params.epistemic_variance_scale,
        params.aleatoric_variance_scale,
    )
    assert np.allclose(post.mu, np.asarray(mu, dtype=float))
    assert np.allclose(post.se, np.asarray(se, dtype=float))
    assert post.idx is not None and idx is not None
    py_idx = np.asarray(post.idx, dtype=int)
    rust_idx = np.asarray(idx, dtype=int)
    assert py_idx.shape == rust_idx.shape
    assert np.array_equal(py_idx, rust_idx)

    seed = 17
    fitter = ENNStatefulFitter(k=4, rng=_Seed(seed))
    rust_fitter = _rust.ENNStatefulFitter(4, seed, True)
    fitter.tell(x, y)
    rust_fitter.tell(x, y)
    fitted = fitter.ask(model, num_fit_candidates=6, num_fit_samples=4)
    rust_fitted = rust_fitter.ask(model.rust_backend, 6, 4)
    assert fitted.k_num_neighbors == rust_fitted.k_num_neighbors
    assert np.allclose(
        fitted.epistemic_variance_scale, rust_fitted.epistemic_variance_scale
    )
    assert np.allclose(
        fitted.aleatoric_variance_scale, rust_fitted.aleatoric_variance_scale
    )
