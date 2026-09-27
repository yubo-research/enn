from __future__ import annotations

import numpy as np
import pytest

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.turbo.config.enn_x_scaling import ENNXScaling
from enn.enn.enn_params import ENNParams

from enn_helpers import enn_all_train_rows


@pytest.mark.parametrize("x_scaling", [ENNXScaling.NONE, ENNXScaling.SCALE_X])
def test_add_rejects_wrong_output_width_without_mutating_model(x_scaling):
    train_x = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=float)
    train_y = np.array([[0.0], [10.0]], dtype=float)
    enn = EpistemicNearestNeighbors(train_x, train_y, x_scaling=x_scaling)

    with pytest.raises(ValueError):
        enn.add(np.array([[20.0, 0.0]], dtype=float), np.array([[20.0, 200.0]]))

    assert len(enn) == 2
    x_at, y_at, _ = enn_all_train_rows(enn)
    np.testing.assert_allclose(x_at, train_x)
    np.testing.assert_allclose(y_at, train_y)

    if x_scaling == ENNXScaling.SCALE_X:
        enn.add(np.array([[30.0, 0.0]], dtype=float), np.array([[30.0]], dtype=float))
        assert len(enn) == 3
        x_at, y_at, _ = enn_all_train_rows(enn)
        assert x_at.shape[0] == y_at.shape[0]

        params = ENNParams(
            k_num_neighbors=1,
            epistemic_variance_scale=1.0,
            aleatoric_variance_scale=0.1,
        )
        out = enn.posterior(np.array([[30.0, 0.0]], dtype=float), params=params)
        np.testing.assert_allclose(out.mu, [[30.0]])
        return

    enn.add(np.array([[30.0, 0.0]], dtype=float), np.array([[30.0]], dtype=float))
    assert len(enn) == 3
    x_at, y_at, _ = enn_all_train_rows(enn)
    assert x_at.shape[0] == y_at.shape[0]

    params = ENNParams(
        k_num_neighbors=1,
        epistemic_variance_scale=1.0,
        aleatoric_variance_scale=0.1,
    )
    out = enn.posterior(np.array([[30.0, 0.0]], dtype=float), params=params)
    np.testing.assert_allclose(out.mu, [[30.0]])


@pytest.mark.parametrize("x_scaling", [ENNXScaling.NONE, ENNXScaling.SCALE_X])
def test_add_first_observations_can_initialize_yvar_on_empty_model(x_scaling):
    empty_x = np.empty((0, 2), dtype=float)
    empty_y = np.empty((0, 1), dtype=float)
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    y = np.array([[10.0], [20.0]], dtype=float)
    yvar = np.array([[0.1], [0.2]], dtype=float)

    incremental = EpistemicNearestNeighbors(empty_x, empty_y, x_scaling=x_scaling)
    incremental.add(x, y, yvar)
    fresh = EpistemicNearestNeighbors(x, y, yvar, x_scaling=x_scaling)

    assert len(incremental) == len(fresh) == 2
    inc_x, inc_y, inc_yvar = enn_all_train_rows(incremental)
    fresh_x, fresh_y, fresh_yvar = enn_all_train_rows(fresh)
    np.testing.assert_allclose(inc_x, fresh_x)
    np.testing.assert_allclose(inc_y, fresh_y)
    np.testing.assert_allclose(inc_yvar, fresh_yvar)


@pytest.mark.parametrize("x_scaling", [ENNXScaling.NONE, ENNXScaling.SCALE_X])
def test_zero_row_add_does_not_change_empty_model_yvar_contract(x_scaling):
    empty_x = np.empty((0, 2), dtype=float)
    empty_y = np.empty((0, 1), dtype=float)
    empty_yvar = np.empty((0, 1), dtype=float)
    x = np.array([[1.0, 2.0]], dtype=float)
    y = np.array([[10.0]], dtype=float)

    incremental = EpistemicNearestNeighbors(empty_x, empty_y, x_scaling=x_scaling)
    incremental.add(empty_x, empty_y, empty_yvar)

    assert len(incremental) == 0
    _, _, yvar = enn_all_train_rows(incremental)
    assert yvar is None

    incremental.add(x, y)
    fresh = EpistemicNearestNeighbors(x, y, x_scaling=x_scaling)

    assert len(incremental) == len(fresh) == 1
    inc_x, inc_y, inc_yvar = enn_all_train_rows(incremental)
    fresh_x, fresh_y, fresh_yvar = enn_all_train_rows(fresh)
    np.testing.assert_allclose(inc_x, fresh_x)
    np.testing.assert_allclose(inc_y, fresh_y)
    assert inc_yvar is None
