"""Controller equation checks do not need Gymnasium or a simulator rollout."""

import numpy as np
import pytest

from evals.short.tlunar import DESCRIPTIONS, NAMES, TurboLunarPolicy, controller_weights


def test_controller_metadata_and_weight_mapping():
    assert len(NAMES) == len(DESCRIPTIONS) == 12
    np.testing.assert_array_equal(controller_weights(-np.ones(12)), np.zeros(12))
    np.testing.assert_array_equal(controller_weights(np.ones(12)), 2 * np.ones(12))
    for value in [np.full(12, np.nan), np.full(12, 1.01), np.zeros(11)]:
        with pytest.raises(ValueError):
            controller_weights(value)


def test_contact_overrides_angle_and_hover_control():
    policy = TurboLunarPolicy()
    policy.set_params(np.zeros(12))
    state = np.array([0, 1, 0, 0, 0, 0, 1, 0], dtype=float)
    assert policy(state) == 0
    state[3] = -2
    assert policy(state) == 2
    state[3] = 0
    weights = np.zeros(12)
    weights[8], weights[11] = 0.5, -0.5
    policy.set_params(weights)
    assert policy(state) == 1
