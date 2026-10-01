"""Read-only view of the Rust AUTO metric on a BPANN_DISK model."""

from __future__ import annotations

import numpy as np

from enn import _rust

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
    """Holds the Rust model that owns the AUTO metric."""

    def __init__(
        self,
        model,
        *,
        rebuild_drift: float | None = None,
        refit_growth: float | None = None,
        tied_dims=(),
        seed: int | None = None,
        reservoir_capacity: int | None = None,
    ) -> None:
        self._inner = model.rust_backend if hasattr(model, "rust_backend") else model
        groups = [list(map(int, g)) for g in tied_dims]
        _rust.metric_configure(
            self._inner,
            None if refit_growth is None else float(refit_growth),
            None if rebuild_drift is None else float(rebuild_drift),
            None if seed is None else int(seed),
            None if reservoir_capacity is None else int(reservoir_capacity),
            groups,
        )

    def _snapshot(self):
        snap = _rust.metric_snapshot(self._inner)
        if snap is None:
            raise RuntimeError("AUTO metric is missing")
        return snap

    def __getattr__(self, name: str):
        try:
            return getattr(self._snapshot(), name)
        except AttributeError:
            raise AttributeError(f"MBPANNMetric has no attribute {name!r}") from None

    @property
    def weights(self) -> np.ndarray:
        return np.asarray(self._snapshot().weights, dtype=float)

    def set_weights(self, weights: np.ndarray) -> bool:
        w = np.asarray(weights, dtype=float)
        return bool(_rust.metric_set_weights(self._inner, w, None))

    def drift(self, weights: np.ndarray) -> float:
        w = np.asarray(weights, dtype=float)
        built = np.asarray(self._snapshot().built, dtype=float)
        return float(_rust.weight_drift(w, built))
