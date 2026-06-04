"""Coverage for examples/try_hnsw_disk.py (kiss per-file gate)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.try_hnsw_disk import main as try_hnsw_disk_main  # noqa: E402


def test_try_hnsw_disk_main():
    try_hnsw_disk_main()
