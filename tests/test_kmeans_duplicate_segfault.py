"""Duplicate observations must not make ennbo's BPANN k-means partition recurse
without shrinking the subproblem. Below ``structured_build_row_limit`` (1024) the
sync path uses a single-leaf build; at 1025 identical rows it enters k-means
partition and must still complete.
"""

from __future__ import annotations

import tempfile

import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams
from enn.turbo.config.enn_index_driver import ENNIndexDriver


STRUCTURED_BUILD_ROW_LIMIT = 1024
DUPLICATE_X = [1.0, 2.0, 3.0]
QUERY = np.array([[1.0, 2.0, 3.0]])


def _enn_params() -> ENNParams:
    return ENNParams(
        k_num_neighbors=5,
        epistemic_variance_scale=1.0,
        aleatoric_variance_scale=0.1,
    )


def _bpann_with_duplicates(n: int, work_dir: str) -> EpistemicNearestNeighbors:
    train_x = np.tile(DUPLICATE_X, (n, 1))
    train_y = np.zeros((n, 1), dtype=float)
    return EpistemicNearestNeighbors(
        train_x,
        train_y,
        scale_x=False,
        index_driver=ENNIndexDriver.BPANN_DISK,
        work_dir=work_dir,
        enn_storage="disk",
    )


def test_duplicate_observations_kmeans_partition_completes() -> None:
    safe_dir = tempfile.mkdtemp()
    safe = _bpann_with_duplicates(STRUCTURED_BUILD_ROW_LIMIT, safe_dir)
    post = safe.posterior(QUERY, params=_enn_params())
    assert post.mu.shape == (1, 1)

    crash_dir = tempfile.mkdtemp()
    large = _bpann_with_duplicates(STRUCTURED_BUILD_ROW_LIMIT + 1, crash_dir)
    post_large = large.posterior(QUERY, params=_enn_params())
    assert post_large.mu.shape == (1, 1)
