from __future__ import annotations

import numpy as np

from enn.turbo.config import ENNFitConfig
from enn.turbo.proposal import mk_enn


def test_mk_enn_empty_returns_none():
    model, params = mk_enn([], [], k=3)
    assert model is None and params is None


def test_mk_enn_builds_model_and_params():
    rng = np.random.default_rng(0)
    x_obs = np.array([[0.0, 0.0], [1.0, 1.0], [0.25, 0.75]])
    y_obs = np.array([0.0, 1.0, 0.2])
    yvar_obs = np.array([0.1, 0.1, 0.1])
    model, params = mk_enn(
        x_obs,
        y_obs,
        k=3,
        yvar_obs=yvar_obs,
        rng=rng,
        fit=ENNFitConfig(num_fit_samples=4, num_fit_candidates=6),
    )
    assert model is not None and params is not None
    assert params.k_num_neighbors == 3
    assert params.epistemic_variance_scale > 0.0


def test_mk_enn_with_fit_config_runs_fitter():
    rng = np.random.default_rng(1)
    x_obs = np.array([[0.0, 0.0], [1.0, 1.0], [0.25, 0.75]], dtype=float)
    y_obs = np.array([0.0, 1.0, 0.2], dtype=float)
    _, params_default = mk_enn(x_obs, y_obs, k=3)
    _, params_fitted = mk_enn(
        x_obs,
        y_obs,
        k=3,
        rng=rng,
        fit=ENNFitConfig(num_fit_samples=4, num_fit_candidates=6),
    )
    assert params_default.epistemic_variance_scale == 1.0
    assert params_default.aleatoric_variance_scale == 0.0
    assert params_fitted.epistemic_variance_scale > 0.0
    assert (
        abs(params_fitted.epistemic_variance_scale - 1.0) > 1e-12
        or abs(params_fitted.aleatoric_variance_scale - 0.0) > 1e-12
    )
