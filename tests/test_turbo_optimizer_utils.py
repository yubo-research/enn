from __future__ import annotations

import numpy as np
import pytest

from enn.turbo.turbo_optimizer_utils import (
    reset_timing,
    sobol_seed_for_state,
    trim_trailing_observations,
    validate_tell_inputs,
)
from enn.turbo.turbo_utils import (
    get_gp_posterior_suppress_warning,
    torch_seed_context,
)
from enn.turbo.types.telemetry import Telemetry


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


def _legacy_keep_indices_when_union_exceeds_cap_buggy(
    incumbent_indices: np.ndarray,
    recent_indices: set[int],
    num_total: int,
    trailing_obs: int,
) -> set[int]:
    """Pre-fix index selection: could leave |keep| > trailing_obs (frozen for regression docs)."""
    keep_indices = set(incumbent_indices.tolist()) | recent_indices
    if len(keep_indices) <= trailing_obs:
        return keep_indices
    keep_indices = set(incumbent_indices.tolist())
    remaining_slots = trailing_obs - len(keep_indices)
    if remaining_slots > 0:
        for i in range(num_total - 1, -1, -1):
            if len(keep_indices) >= trailing_obs:
                break
            if i not in keep_indices:
                keep_indices.add(i)
    return keep_indices


def test_legacy_trim_index_logic_would_exceed_cap_with_many_incumbents():
    """This assertion **passes**: it records that old logic violated the cap (would fail production)."""
    n = 30
    trailing = 5
    start_idx = max(0, n - trailing)
    recent = set(range(start_idx, n))
    legacy = _legacy_keep_indices_when_union_exceeds_cap_buggy(
        np.arange(20, dtype=int), recent, n, trailing
    )
    assert len(legacy) > trailing


def test_trim_trailing_observations_respects_cap_when_many_incumbents():
    """Regression: production must never return more than trailing_obs rows."""
    n = 30
    trailing = 5
    x_list = [[i] for i in range(n)]
    y_list = [[float(i)] for i in range(n)]
    y_tr = [float(i) for i in range(n)]
    yvar: list[float] = []
    incumbent = np.arange(20, dtype=int)
    result = trim_trailing_observations(
        x_list, y_list, y_tr, yvar, trailing_obs=trailing, incumbent_indices=incumbent
    )
    assert len(result.x_obs) <= trailing


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
    import torch

    from enn.turbo.turbo_gp_fit import fit_gp

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
