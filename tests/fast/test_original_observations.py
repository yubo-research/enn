"""Observation read-back preserves supplied values through feature and output warps."""

import numpy as np
import pytest

from enn import EpistemicNearestNeighbors, SemanticInputTransform
from enn.turbo.config.enn_index_driver import ENNIndexDriver


class Encoder:
    def encode(self, texts):
        return np.eye(len(texts))


@pytest.mark.parametrize("semantic_only", [False, True])
@pytest.mark.parametrize("disk", [False, True])
def test_exact_rows_survive_append_and_disk_reopen(tmp_path, disk, semantic_only):
    transform = SemanticInputTransform(
        ["a", "b"], [[-3, 7], [100, 1000]], encoder=Encoder()
    )
    x = np.array(
        [[0.1, 123.4], [np.nextafter(0.0, 1.0), 789.123], [6.123456789, 500.1]]
    )
    y = np.array([[0.1], [0.9], [np.nextafter(0.5, 1.0)]])
    kwargs = (
        {"index_driver": ENNIndexDriver.BPANN_DISK, "work_dir": tmp_path}
        if disk
        else {}
    )
    model = EpistemicNearestNeighbors(
        x[:2],
        y[:2],
        input_transform=transform,
        semantic_only=semantic_only,
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
        model = EpistemicNearestNeighbors(
            np.empty((0, 2)),
            np.empty((0, 1)),
            input_transform=transform,
            semantic_only=semantic_only,
            **kwargs,
        )
        reopened_x, reopened_y, _ = model.train_rows_at(indices)
        np.testing.assert_array_equal(reopened_x, actual_x)
        np.testing.assert_array_equal(reopened_y, actual_y)
