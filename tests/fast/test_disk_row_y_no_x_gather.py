"""row_y must not materialize train_x (RSS regression for large disk N)."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.turbo.config.enn_index_driver import ENNIndexDriver


def _live_rss_bytes() -> int:
    out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())])
    return int(out.decode().strip()) * 1024


@pytest.mark.skipif(
    sys.platform != "darwin", reason="ps RSS probe calibrated on darwin"
)
def test_disk_row_y_iteration_does_not_fault_full_train_x():
    """Iterating row_y over all rows must not charge ~N·D·8 bytes of train_x."""
    n = 100_000
    d = 10
    train_x_mib = n * d * 8 / (1024 * 1024)
    with tempfile.TemporaryDirectory(prefix="_enn_row_y_") as work:
        model = EpistemicNearestNeighbors(
            np.empty((0, d)),
            np.empty((0, 1)),
            scale_x=False,
            index_driver=ENNIndexDriver.BPANN_DISK,
            work_dir=work,
            enn_storage="disk",
        )
        rng = np.random.default_rng(0)
        x = rng.normal(size=(n, d))
        y = rng.normal(size=(n, 1))
        model.add(x, y)
        model.ensure_index_sync()
        before = _live_rss_bytes()
        got = np.array(
            [
                float(np.asarray(model._rust_model.row_y(i)).reshape(-1)[0])
                for i in range(n)
            ],
            dtype=float,
        )
        after = _live_rss_bytes()
        delta_mib = (after - before) / (1024 * 1024)
        np.testing.assert_allclose(got, y[:, 0], rtol=0, atol=0)
        assert delta_mib < 0.55 * train_x_mib, (
            f"row_y iteration charged {delta_mib:.1f} MiB; train_x is {train_x_mib:.1f} MiB"
        )
