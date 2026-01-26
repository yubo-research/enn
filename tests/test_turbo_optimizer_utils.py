from __future__ import annotations
import numpy as np
import pytest
from enn.turbo.turbo_optimizer_utils import (
    reset_timing,
    sobol_seed_for_state,
    validate_tell_inputs,
    trim_trailing_observations,
)
from enn.turbo.types.telemetry import Telemetry
from enn.turbo.turbo_utils import (
    torch_seed_context,
    get_gp_posterior_suppress_warning,
)


def test_sobol_seed_for_state_deterministic():
    result1 = sobol_seed_for_state(12345, restart_generation=0, n_obs=10, num_arms=4)
    result2 = sobol_seed_for_state(12345, restart_generation=0, n_obs=10, num_arms=4)
    assert result1 == result2


def test_sobol_seed_for_state_changes_with_inputs():
    base = sobol_seed_for_state(12345, restart_generation=0, n_obs=10, num_arms=4)
    diff_seed = sobol_seed_for_state(54321, restart_generation=0, n_obs=10, num_arms=4)
    diff_restart = sobol_seed_for_state(
        12345, restart_generation=1, n_obs=10, num_arms=4
    )
    diff_obs = sobol_seed_for_state(12345, restart_generation=0, n_obs=20, num_arms=4)
    diff_arms = sobol_seed_for_state(12345, restart_generation=0, n_obs=10, num_arms=8)
    assert (
        base != diff_seed
        and base != diff_restart
        and base != diff_obs
        and base != diff_arms
    )


def test_validate_tell_inputs_valid_2d():
    x = np.random.randn(10, 3)
    y = np.random.randn(10, 2)
    y_var = np.random.rand(10, 2)
    result = validate_tell_inputs(x, y, y_var, num_dim=3)
    assert result.x.shape == (10, 3) and result.y.shape == (10, 2)
    assert result.y_var.shape == (10, 2) and result.num_metrics == 2


def test_validate_tell_inputs_valid_1d():
    x = np.random.randn(10, 3)
    y = np.random.randn(10)
    result = validate_tell_inputs(x, y, None, num_dim=3)
    assert (
        result.x.shape == (10, 3)
        and result.y.shape == (10,)
        and result.num_metrics == 1
    )


def test_validate_tell_inputs_invalid_x_shape():
    with pytest.raises(ValueError):
        validate_tell_inputs(
            np.random.randn(10, 4), np.random.randn(10), None, num_dim=3
        )


def test_validate_tell_inputs_mismatched_y():
    with pytest.raises(ValueError):
        validate_tell_inputs(
            np.random.randn(10, 3), np.random.randn(5), None, num_dim=3
        )


def test_validate_tell_inputs_invalid_y_shape():
    with pytest.raises(ValueError):
        validate_tell_inputs(
            np.random.randn(10, 3), np.random.randn(10, 2, 3), None, num_dim=3
        )


def test_trim_trailing_observations_no_trim_needed():
    x_list = [[0.1, 0.2]] * 5
    y_list = [1.0] * 5
    y_tr = [1.0] * 5
    yvar = [0.1] * 5
    incumbent = np.array([0])
    result = trim_trailing_observations(
        x_list, y_list, y_tr, yvar, trailing_obs=10, incumbent_indices=incumbent
    )
    assert len(result.x_obs) == 5


def test_trim_trailing_observations_trims():
    x_list = [[i, i] for i in range(20)]
    y_list = list(range(20))
    y_tr = list(range(20))
    yvar = [0.1] * 20
    incumbent = np.array([0])
    result = trim_trailing_observations(
        x_list, y_list, y_tr, yvar, trailing_obs=5, incumbent_indices=incumbent
    )
    assert len(result.x_obs) <= 5 and 0 in [row[0] for row in result.x_obs]


def test_telemetry_dataclass():
    t = Telemetry(dt_fit=0.5, dt_sel=0.3)
    assert t.dt_fit == 0.5 and t.dt_sel == 0.3


@pytest.mark.parametrize("seed1,seed2,should_match", [(42, 42, True), (42, 43, False)])
def test_torch_seed_context(seed1, seed2, should_match):
    import torch

    with torch_seed_context(seed1):
        val1 = torch.randn(3).tolist()
    with torch_seed_context(seed2):
        val2 = torch.randn(3).tolist()
    assert (val1 == val2) == should_match


def test_get_gp_posterior_suppress_warning_basic():
    from enn.turbo.turbo_gp_fit import fit_gp
    import torch

    x = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]
    y = [1.0, 2.0, 3.0, 4.0]
    gp_result = fit_gp(x, y, num_dim=2, num_steps=10)
    if gp_result.model is not None:
        x_torch = torch.tensor([[0.2, 0.3]], dtype=torch.float64)
        result = get_gp_posterior_suppress_warning(gp_result.model, x_torch)
        assert result is not None


def test_reset_timing():
    class Obj:
        _dt_fit = 1.0
        _dt_gen = 2.0
        _dt_sel = 3.0

    o = Obj()
    reset_timing(o)
    assert o._dt_fit == 0.0 and o._dt_gen == 0.0 and o._dt_sel == 0.0
