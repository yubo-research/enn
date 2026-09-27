"""Incremental metric updates for an ENN opened with ``index_driver=MBPANN_DISK``.

Metric weights ``w`` define the ENN distance ``sum_d w_d (x_d - x'_d)^2``. The
BPANN index stores coordinates ``x_d * sqrt(w_d)``, so a new ``w`` can be applied
to every stored coordinate exactly without re-reading rows; only the partition
(which rows share a leaf) goes stale. ``MBPANNMetric`` rescales in place and
re-partitions only when the metric has drifted far from the one the partition
was built under.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from enn.turbo.config.enn_index_driver import ENNIndexDriver

if TYPE_CHECKING:
    from .enn_class import EpistemicNearestNeighbors

DEFAULT_REBUILD_DRIFT = float(np.log(2.0))


class MBPANNMetric:
    """Owns the metric of an MBPANN_DISK model.

    ``rebuild_drift`` is the largest per-dimension change of the distance scale,
    ``max_d |log(sqrt(w_d / w_built_d))|``, tolerated before re-partitioning.
    ``0`` re-partitions on every change; ``inf`` never does.
    """

    def __init__(
        self,
        model: EpistemicNearestNeighbors,
        *,
        rebuild_drift: float = DEFAULT_REBUILD_DRIFT,
    ) -> None:
        if model._index_driver != ENNIndexDriver.MBPANN_DISK:
            raise ValueError("MBPANNMetric requires index_driver=MBPANN_DISK")
        if not rebuild_drift >= 0:
            raise ValueError(f"rebuild_drift must be >= 0, got {rebuild_drift}")
        num_dim = model._num_dim
        self._model = model
        self._rebuild_drift = float(rebuild_drift)
        self._weights = np.ones(num_dim)
        self._built_weights = np.ones(num_dim)
        self.num_rescales = 0
        self.num_rebuilds = 0

    @property
    def weights(self) -> np.ndarray:
        return self._weights.copy()

    def _validated(self, weights: np.ndarray) -> np.ndarray:
        w = np.asarray(weights, dtype=float)
        if w.shape != self._weights.shape:
            raise ValueError(f"weights must have shape {self._weights.shape}, got {w.shape}")
        if not np.all(np.isfinite(w) & (w > 0)):
            raise ValueError("weights must be finite and > 0")
        return w

    def drift(self, weights: np.ndarray) -> float:
        """Scale drift of ``weights`` from the metric of the last re-partition."""
        w = self._validated(weights)
        return float(0.5 * np.max(np.abs(np.log(w / self._built_weights))))

    def set_weights(self, weights: np.ndarray) -> bool:
        """Apply new metric weights; return True if the index was re-partitioned."""
        w = self._validated(weights)
        rebuild = self.drift(w) > self._rebuild_drift
        self._model.rust_backend.set_metric_scale(1.0 / np.sqrt(w), rebuild=rebuild)
        self._weights = w
        if rebuild:
            self._built_weights = w
            self.num_rebuilds += 1
        else:
            self.num_rescales += 1
        return rebuild
