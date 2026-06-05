"""Disk HNSW background flush contract for EpistemicNearestNeighbors.add (Python)."""

import json
import threading

import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.turbo.config.enn_index_driver import ENNIndexDriver


def _read_indexed_rows(work_dir) -> int:
    text = (work_dir / "metadata.json").read_text()
    return json.loads(text)["indexed_rows"]


def test_py_model_add_waits_for_flush(tmp_path):
    work_dir = tmp_path / "disk_flush_add"
    rng = np.random.default_rng(3)
    d = 2
    model = EpistemicNearestNeighbors(
        np.zeros((0, d)),
        np.zeros((0, 1)),
        index_driver=ENNIndexDriver.HNSW_DISK,
        work_dir=str(work_dir),
        enn_storage="disk",
    )
    init_x = rng.standard_normal((1000, d))
    init_y = rng.standard_normal((1000, 1))
    model.add(init_x, init_y)
    model.ensure_index_sync()
    assert _read_indexed_rows(work_dir) == 1000

    extra_x = rng.standard_normal((1000, d))
    extra_y = rng.standard_normal((1000, 1))
    model.add(extra_x, extra_y)
    assert len(model) == 2000
    assert _read_indexed_rows(work_dir) < len(model)

    model.schedule_background_flush()
    add_err = []

    row_x = rng.standard_normal((1, d))
    row_y = rng.standard_normal((1, 1))

    def add_one():
        try:
            model.add(row_x, row_y)
        except Exception as exc:  # pragma: no cover - failure path
            add_err.append(exc)

    t = threading.Thread(target=add_one)
    t.start()
    t.join()
    assert not add_err

    model.ensure_index_sync()
    assert _read_indexed_rows(work_dir) == len(model)
