"""Observation read-back preserves supplied values through output warps."""

import numpy as np
import pytest

from enn import EpistemicNearestNeighbors


@pytest.mark.parametrize("disk", [False, True])
def test_exact_rows_survive_append_and_disk_reopen(tmp_path, disk):
    x = np.array(
        [[0.1, 123.4], [np.nextafter(0.0, 1.0), 789.123], [6.123456789, 500.1]]
    )
    y = np.array([[0.1], [0.9], [np.nextafter(0.5, 1.0)]])
    kwargs = {"work_dir": tmp_path}
    model = EpistemicNearestNeighbors(
        x[:2],
        y[:2],
        y_bounds=np.array([[0.0, 1.0]]),
        **kwargs,
    )
    model.add(x[2:], y[2:])
    indices = [2, 0, 1, 2]
    actual_x, actual_y, _ = model.train_rows_at(indices)
    np.testing.assert_array_equal(actual_x, x[indices])
    np.testing.assert_array_equal(actual_y, y[indices])
    x[:] = 0
    y[:] = 0
    np.testing.assert_array_equal(model.train_rows_at(indices)[0], actual_x)
    if disk:
        model.persist_index_to_disk()
        del model
        model = EpistemicNearestNeighbors.open(kwargs["work_dir"])
        reopened_x, reopened_y, _ = model.train_rows_at(indices)
        np.testing.assert_array_equal(reopened_x, actual_x)
        np.testing.assert_array_equal(reopened_y, actual_y)
