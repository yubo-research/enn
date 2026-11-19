from __future__ import annotations

import numpy as np
import pytest

from enn.enn.enn_class import EpistemicNearestNeighbors
from evals.harness.work_dirs import fresh_work_dir


def _model() -> EpistemicNearestNeighbors:
    rng = np.random.default_rng(0)
    return EpistemicNearestNeighbors(
        rng.random((6, 2)), rng.random((6, 1)), work_dir=fresh_work_dir()
    )


def test_add_with_wrong_x_width_reports_the_x_width_expected():
    with pytest.raises(ValueError, match=r"expected \[1, 2\], got \[1, 3\]"):
        _model().add(np.zeros((1, 3)), np.zeros((1, 1)))


def test_add_with_wrong_y_width_reports_the_metric_columns():
    with pytest.raises(ValueError, match="metric columns"):
        _model().add(np.zeros((1, 2)), np.zeros((1, 2)))
