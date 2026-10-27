"""Behavioral contracts for the controller relationship experiment."""

import numpy as np
import pytest

from _reports.semantic_features import eval_lunar_benchmark
from _reports.semantic_features import tlunar_interaction_report as report
from _reports.semantic_features.eval_tlunar_interactions import (
    CORNERS,
    SCALE_GROUPS,
    action_vector,
    mixed_difference,
    scaled_point,
    state_bank,
)
from _reports.semantic_features.tlunar import TurboLunarPolicy


def test_common_command_scale_preserves_flight_and_contact_actions():
    rng = np.random.default_rng(312)
    states = state_bank(rng, count=64)
    for point in rng.uniform(-0.8, 0.2, (8, 12)):
        expected = action_vector(point, states)
        for factor in (0.5, 1.5):
            scaled = scaled_point(
                point, SCALE_GROUPS["all_commands_and_thresholds"], factor
            )
            np.testing.assert_array_equal(action_vector(scaled, states), expected)
            np.testing.assert_allclose(scaled[:4], point[:4])


def test_branch_scales_change_engine_arbitration_at_ties():
    point = np.zeros(12)
    state = np.array([0, -2, 0, 0, -1, 0, 0, 0], dtype=float)
    policy = TurboLunarPolicy()
    policy.set_params(point)
    assert policy(state) == 2
    for group, factor in (
        ("angle_only", 2),
        ("hover_only", 0.5),
        ("thresholds_only", 2),
    ):
        policy.set_params(scaled_point(point, SCALE_GROUPS[group], factor))
        assert policy(state) == 0
    policy.set_params(
        scaled_point(point, SCALE_GROUPS["all_commands_and_thresholds"], 2)
    )
    assert policy(state) == 2
    state[6] = 1
    state[3] = -2
    assert policy(state) == 2


def test_mixed_contrast_rejects_product_and_accepts_additive_objectives():
    corners = np.array(CORNERS)
    additive = 3 * corners[:, 0] - 2 * corners[:, 1] + 8
    np.testing.assert_array_equal(mixed_difference([additive, additive]), [0, 0])
    assert mixed_difference(corners[:, 0] * corners[:, 1]) == 1


@pytest.mark.parametrize("factor", [0, -1, np.nan, np.inf, 3])
def test_scale_interventions_reject_invalid_weights(factor):
    with pytest.raises(ValueError):
        scaled_point(np.zeros(12), range(4, 12), factor)


def test_historical_projection_cannot_silently_use_revised_names(tmp_path, monkeypatch):
    from enn import SemanticInputTransform
    from tests.fast.test_semantic import Encoder

    transform = SemanticInputTransform(["old_name"], [[-1, 1]], encoder=Encoder())
    transform.save(tmp_path / "lunar_projection.npz")
    monkeypatch.setattr(eval_lunar_benchmark, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="historical parameter names"):
        eval_lunar_benchmark.lunar_projection()


def test_report_preserves_interactions_that_cancel_across_simulator_seeds():
    product = np.prod(np.array(CORNERS), axis=1)
    returns = np.tile(np.array([product, -product])[None, None], (4, 1, 1, 1))
    rows = report.interaction_summary(returns, [(0, 4)])
    assert rows == [
        {
            "i": 0,
            "j": 4,
            "rms_contrast": 1.0,
            "ci_low": 1.0,
            "ci_high": 1.0,
            "nonzero_fraction": 1.0,
        }
    ]
