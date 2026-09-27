"""Incremental metric updates for an ENN opened with ``index_driver=BPANN_DISK`` and
``x_scaling=ENNXScaling.METRIC_LEARNING``.

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

from enn.turbo.config.enn_x_scaling import ENNXScaling

if TYPE_CHECKING:
    from .enn_class import EpistemicNearestNeighbors

DEFAULT_REBUILD_DRIFT = float(np.log(2.0))
DRIFT_WEIGHT_FLOOR = float(np.log(1e4))


def _floored_log(w: np.ndarray) -> np.ndarray:
    log_w = np.log(w)
    return np.maximum(log_w, log_w.max() - DRIFT_WEIGHT_FLOOR)


class MBPANNMetric:
    """Owns the metric of a metric-learning BPANN_DISK model.

    ``rebuild_drift`` is the largest per-dimension change of the distance scale,
    ``max_d |log(sqrt(w_d / w_built_d))|``, tolerated before re-partitioning.
    ``0`` re-partitions on every change; ``inf`` never does. Before comparing, each
    ``log w`` is raised to at least ``max(log w) - DRIFT_WEIGHT_FLOOR``: a dimension
    weighted 1e4 times less than the heaviest one barely moves distances, so its
    wandering does not trigger a re-partition.
    """

    def __init__(
        self,
        model: EpistemicNearestNeighbors,
        *,
        rebuild_drift: float = DEFAULT_REBUILD_DRIFT,
    ) -> None:
        if model.x_scaling != ENNXScaling.METRIC_LEARNING:
            raise ValueError("MBPANNMetric requires x_scaling=METRIC_LEARNING")
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
        return float(0.5 * np.max(np.abs(_floored_log(w) - _floored_log(self._built_weights))))

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
