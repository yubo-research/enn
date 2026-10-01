from __future__ import annotations

import numpy as np
import pytest

from enn.turbo.config.enn_index_driver import ENNIndexDriver
from enn.turbo.config.enn_surrogate_config import ENNStorage


@pytest.mark.slow
def test_disk_persisted_store_10k_reopens_fast(tmp_path):
    """After persist_index_to_disk, 10k-row reopen init should be mmap-fast."""
    import time

    from enn.enn.enn_class import EpistemicNearestNeighbors
    from ops.stress import make_synthetic_observations

    work_dir = tmp_path / "enn_persist_reopen_10k"
    num_obs = 10_000
    num_dim = 32
    model = EpistemicNearestNeighbors(
        np.empty((0, num_dim)),
        np.empty((0, 1)),
        index_driver=ENNIndexDriver.BPANN_DISK,
        work_dir=str(work_dir),
        enn_storage=ENNStorage.DISK,
    )
    batch = 500
    for start in range(0, num_obs, batch):
        end = min(start + batch, num_obs)
        xs, ys = make_synthetic_observations(
            end - start, num_dim=num_dim, seed=0 + start
        )
        model.add(xs, ys)
        model.schedule_background_flush()
    model.persist_index_to_disk()

    t0 = time.perf_counter()
    EpistemicNearestNeighbors(
        np.empty((0, num_dim)),
        np.empty((0, 1)),
        index_driver=ENNIndexDriver.BPANN_DISK,
        work_dir=str(work_dir),
        enn_storage=ENNStorage.DISK,
    )
    init_s = time.perf_counter() - t0
    assert init_s < 1.0, f"reopen init_s={init_s:.3f}s expected < 1.0s after persist"
