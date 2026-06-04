"""Coverage for examples/try_hannoy.py (kiss per-file gate)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples import try_hannoy  # noqa: E402
from examples import hannoy_smoke_runner  # noqa: E402


def test_brute_knn_matches_numpy_argsort():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((20, 4), dtype=np.float32)
    query = rng.standard_normal(4, dtype=np.float32)
    got = try_hannoy.brute_knn(query, data, k=3)
    dists = np.sum((data - query) ** 2, axis=1)
    expected = [(int(i), float(dists[i])) for i in np.argsort(dists)[:3]]
    assert got == expected


def test_try_hannoy_main_smoke():
    mock_db = MagicMock()
    mock_reader = MagicMock()
    mock_reader.by_vec.return_value = [(0, 0.0)]
    mock_db.reader.return_value = mock_reader
    mock_writer = MagicMock()
    mock_db.writer.return_value.__enter__ = MagicMock(return_value=mock_writer)
    mock_db.writer.return_value.__exit__ = MagicMock(return_value=False)

    mock_hannoy = MagicMock()
    mock_hannoy.Database.return_value = mock_db
    mock_hannoy.Metric = MagicMock()

    with patch.dict(sys.modules, {"hannoy": mock_hannoy}):
        hannoy_smoke_runner.run_hannoy_smoke()
