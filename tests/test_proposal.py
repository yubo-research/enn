from __future__ import annotations
import numpy as np
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
    )
    assert model is not None and params is not None
    assert params.k_num_neighbors == 3
