"""Incremental metric updates for an ENN opened with ``index_driver=BPANN_DISK`` and
``metric_learning=ENNMetricLearning.AUTO``.

Metric weights ``w`` define the ENN distance ``sum_d w_d (x_d - x'_d)^2``. The
BPANN index stores coordinates ``x_d * sqrt(w_d)``, so a new ``w`` can be applied
to every stored coordinate exactly without re-reading rows; only the partition
(which rows share a leaf) goes stale. ``MBPANNMetric`` rescales in place and
re-partitions only when the metric has drifted far from the one the partition
was built under.
"""

from __future__ import annotations

import math
from typing import Any, Protocol

import numpy as np

from enn.turbo.config.enn_x_scaling import ENNMetricLearning

from .metric_stream import MIN_DEPENDENCE_ROWS, Reservoir, auto_weights

DEFAULT_REBUILD_DRIFT = float(np.log(2.0))
DRIFT_WEIGHT_FLOOR = float(np.log(1e4))
AUTO_MIN_HELDOUT_GAIN = 0.0
AUTO_RESERVOIR_CAPACITY = 1000
AUTO_K = 10
AUTO_REFIT_GROWTH = 1.5
AUTO_RESCALE_TOL = 0.01


class _MetricModel(Protocol):
    """The parts of ``EpistemicNearestNeighbors`` that ``MBPANNMetric`` uses."""

    @property
    def metric_learning(self) -> ENNMetricLearning: ...

    @property
    def rust_backend(self) -> Any: ...

    @property
    def _num_dim(self) -> int: ...

    @property
    def _num_metrics(self) -> int: ...


def auto_uses_learned_metric(heldout_gain: float) -> bool:
    """The ``ENNMetricLearning.AUTO`` rule: use the learned metric only if it validated better.

    ``heldout_gain`` is the mean per-row leave-one-out log-likelihood of the learned diagonal
    metric minus that of the best isotropic metric (``metric_stream.auto_weights``). Each row is
    scored from neighbors that exclude it, so an input that only fits noise does not pay off.
    """
    return bool(np.isfinite(heldout_gain) and heldout_gain > AUTO_MIN_HELDOUT_GAIN)


def _floored_log(w: np.ndarray) -> np.ndarray:
    log_w = np.log(w)
    return np.maximum(log_w, log_w.max() - DRIFT_WEIGHT_FLOOR)


class MBPANNMetric:
    """Owns the metric of an AUTO BPANN_DISK model.

    ``rebuild_drift`` is the largest per-dimension change of the distance scale,
    ``max_d |log(sqrt(w_d / w_built_d))|``, tolerated before re-partitioning.
    ``0`` re-partitions on every change; ``inf`` never does. Before comparing, each
    ``log w`` is raised to at least ``max(log w) - DRIFT_WEIGHT_FLOOR``: a dimension
    weighted 1e4 times less than the heaviest one barely moves distances, so its
    wandering does not trigger a re-partition.

    ``observe`` offers added rows to a reservoir of ``reservoir_capacity`` rows. Once
    ``MIN_DEPENDENCE_ROWS`` rows have been seen, and again whenever the number seen has grown
    by ``refit_growth`` since the last refit, ``refit`` computes ``auto_weights`` on the
    reservoir and applies them if ``auto_uses_learned_metric``, else the identity metric.
    A change smaller than ``AUTO_RESCALE_TOL`` in every distance scale is not applied.
    """

    def __init__(
        self,
        model: _MetricModel,
        *,
        rebuild_drift: float = DEFAULT_REBUILD_DRIFT,
        reservoir_capacity: int = AUTO_RESERVOIR_CAPACITY,
        refit_growth: float = AUTO_REFIT_GROWTH,
        seed: int = 0,
    ) -> None:
        if model.metric_learning != ENNMetricLearning.AUTO:
            raise ValueError("MBPANNMetric requires metric_learning=AUTO")
        if not rebuild_drift >= 0:
            raise ValueError(f"rebuild_drift must be >= 0, got {rebuild_drift}")
        if not refit_growth > 1:
            raise ValueError(f"refit_growth must be > 1, got {refit_growth}")
        num_dim = model._num_dim
        self._model = model
        self._rebuild_drift = float(rebuild_drift)
        self._refit_growth = float(refit_growth)
        self._weights = np.ones(num_dim)
        self._built_weights = np.ones(num_dim)
        self.reservoir = Reservoir(
            reservoir_capacity, num_dim, np.random.default_rng(seed), model._num_metrics
        )
        self._next_refit = MIN_DEPENDENCE_ROWS
        self.heldout_gain: float | None = None
        self.num_refits = 0
        self.num_rescales = 0
        self.num_rebuilds = 0

    @property
    def weights(self) -> np.ndarray:
        return self._weights.copy()

    @property
    def uses_learned_metric(self) -> bool:
        return self.heldout_gain is not None and auto_uses_learned_metric(self.heldout_gain)

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
        """Apply new metric weights; return True if the index was re-partitioned.

        The next ``refit`` replaces them.
        """
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

    def observe(self, x: np.ndarray, y: np.ndarray) -> None:
        """Offer added rows to the reservoir; ``refit`` if the row count reached the next refit point."""
        self.reservoir.add(x, y)
        if self.reservoir.num_seen >= self._next_refit:
            self.refit()

    def refit(self) -> None:
        w, gain = auto_weights(self.reservoir.x, self.reservoir.y, AUTO_K)
        self.heldout_gain = gain
        self.num_refits += 1
        self._next_refit = max(MIN_DEPENDENCE_ROWS, math.ceil(self._refit_growth * self.reservoir.num_seen))
        target = w if auto_uses_learned_metric(gain) else np.ones_like(self._weights)
        if 0.5 * np.max(np.abs(np.log(target / self._weights))) > AUTO_RESCALE_TOL:
            self.set_weights(target)
