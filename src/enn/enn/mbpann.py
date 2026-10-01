"""Read-only view of the Rust AUTO metric on a BPANN_DISK model."""

from __future__ import annotations

import numpy as np

from enn import _rust
from enn.turbo.config.enn_x_scaling import ENNMetricLearning

DEFAULT_REBUILD_DRIFT = float(_rust.DEFAULT_REBUILD_DRIFT)
DRIFT_WEIGHT_FLOOR = float(_rust.DRIFT_WEIGHT_FLOOR)
AUTO_MIN_HELDOUT_GAIN = float(_rust.AUTO_MIN_HELDOUT_GAIN)
AUTO_RESERVOIR_CAPACITY = int(_rust.AUTO_RESERVOIR_CAPACITY)
AUTO_K = int(_rust.AUTO_K)
AUTO_REFIT_GROWTH = float(_rust.AUTO_REFIT_GROWTH)
AUTO_RESCALE_TOL = float(_rust.AUTO_RESCALE_TOL)


def auto_uses_learned_metric(heldout_gain: float) -> bool:
    return bool(_rust.auto_uses_learned_metric(float(heldout_gain)))


class MBPANNMetric:
    """Forwards metric state to the Rust model that owns the policy."""

    def __init__(
        self,
        model,
        *,
        rebuild_drift: float = DEFAULT_REBUILD_DRIFT,
        refit_growth: float = AUTO_REFIT_GROWTH,
        tied_dims=(),
        seed: int = 0,
        reservoir_capacity: int = AUTO_RESERVOIR_CAPACITY,
    ) -> None:
        if model.metric_learning != ENNMetricLearning.AUTO:
            raise ValueError("MBPANNMetric requires metric_learning=AUTO")
        if seed < 0:
            raise ValueError(f"seed must be >= 0, got {seed}")
        groups = [list(map(int, g)) for g in tied_dims]
        stored = [list(map(int, g)) for g in _rust.metric_tied(model.rust_backend)]
        if groups != stored:
            raise ValueError(
                f"tied_dims {groups} do not match the model groups {stored}"
            )
        _rust.metric_configure(
            model.rust_backend,
            float(refit_growth),
            float(rebuild_drift),
            int(seed),
            int(reservoir_capacity),
        )
        self._model = model
        self._rebuild_drift = float(rebuild_drift)

    def _snapshot(self):
        snap = _rust.metric_snapshot(self._model.rust_backend)
        if snap is None:
            raise RuntimeError("AUTO metric is missing")
        return snap

    def __getattr__(self, name: str):
        try:
            return getattr(self._snapshot(), name)
        except AttributeError:
            raise AttributeError(
                f"MBPANNMetric has no attribute {name!r}"
            ) from None

    @property
    def weights(self) -> np.ndarray:
        return np.asarray(self._snapshot().weights, dtype=float)

    def set_weights(self, weights: np.ndarray) -> bool:
        w = np.asarray(weights, dtype=float)
        if w.shape != (self._model._num_dim,):
            raise ValueError(
                f"weights must have shape {(self._model._num_dim,)}, got {w.shape}"
            )
        if not np.all(np.isfinite(w) & (w > 0)):
            raise ValueError("weights must be finite and > 0")
        return bool(
            _rust.metric_set_weights(self._model.rust_backend, w, self._rebuild_drift)
        )

    def drift(self, weights: np.ndarray) -> float:
        w = np.asarray(weights, dtype=float)
        built = np.asarray(self._snapshot().built, dtype=float)
        if w.shape != built.shape:
            raise ValueError(f"weights shape {w.shape} != built shape {built.shape}")
        return float(_rust.weight_drift(w, built))
