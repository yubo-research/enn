"""Read-only view of the Rust AUTO metric on a BPANN_DISK model."""

from __future__ import annotations

import numpy as np

from enn import _rust
from enn.turbo.config.enn_x_scaling import ENNMetricLearning

DEFAULT_REBUILD_DRIFT = float(np.log(2.0))
DRIFT_WEIGHT_FLOOR = float(np.log(1e4))
AUTO_MIN_HELDOUT_GAIN = 0.0
AUTO_RESERVOIR_CAPACITY = 1000
AUTO_K = 10
AUTO_REFIT_GROWTH = 1.5
AUTO_RESCALE_TOL = 0.01


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
        if not rebuild_drift >= 0:
            raise ValueError(f"rebuild_drift must be >= 0, got {rebuild_drift}")
        if not refit_growth > 1:
            raise ValueError(f"refit_growth must be > 1, got {refit_growth}")
        if seed < 0:
            raise ValueError(f"seed must be >= 0, got {seed}")
        if reservoir_capacity < 1:
            raise ValueError(
                f"reservoir_capacity must be >= 1, got {reservoir_capacity}"
            )
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

    @property
    def weights(self) -> np.ndarray:
        w = _rust.metric_weights(self._model.rust_backend)
        return (
            np.ones(self._model._num_dim) if w is None else np.asarray(w, dtype=float)
        )

    @property
    def heldout_gain(self):
        return _rust.metric_heldout_gain(self._model.rust_backend)

    @property
    def num_seen(self) -> int:
        return int(_rust.metric_num_seen(self._model.rust_backend))

    @property
    def num_refits(self) -> int:
        return int(_rust.metric_num_refits(self._model.rust_backend))

    @property
    def num_rescales(self) -> int:
        return int(_rust.metric_num_rescales(self._model.rust_backend))

    @property
    def num_rebuilds(self) -> int:
        return int(_rust.metric_num_rebuilds(self._model.rust_backend))

    @property
    def uses_learned_metric(self) -> bool:
        return bool(_rust.metric_uses_learned(self._model.rust_backend))

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
        built_w = _rust.metric_built(self._model.rust_backend)
        built = self.weights if built_w is None else np.asarray(built_w, dtype=float)
        if w.shape != built.shape:
            raise ValueError(f"weights shape {w.shape} != built shape {built.shape}")
        return float(_rust.weight_drift(w, built))
