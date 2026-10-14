from __future__ import annotations

import numpy as np

from enn._rust import ENNTS as _RustENNTS

_CONFIG_DEFAULTS = {
    "horizon": 1,
    "cadence": 1,
    "num_tables": 8,
    "num_projections": 4,
    "bucket_width": 4.0,
    "k": 8,
    "probe": 1,
    "seed": 0,
    "epistemic_variance_scale": 1.0,
    "aleatoric_variance_scale": 0.1,
    "metric_learning": "none",
}


class ENNTSConfig:
    def __init__(self, lookback: int, **overrides: object) -> None:
        unknown = set(overrides) - set(_CONFIG_DEFAULTS)
        if unknown:
            names = ", ".join(sorted(str(name) for name in unknown))
            raise ValueError(f"unknown ENNTSConfig fields: {names}")
        values = dict(_CONFIG_DEFAULTS)
        values.update(overrides)
        self.lookback = int(lookback)
        self.horizon = int(values["horizon"])
        self.cadence = int(values["cadence"])
        self.num_tables = int(values["num_tables"])
        self.num_projections = int(values["num_projections"])
        self.bucket_width = float(values["bucket_width"])
        self.k = int(values["k"])
        self.probe = int(values["probe"])
        self.seed = int(values["seed"])
        self.epistemic_variance_scale = float(values["epistemic_variance_scale"])
        self.aleatoric_variance_scale = float(values["aleatoric_variance_scale"])
        metric = str(values["metric_learning"]).lower()
        if metric not in {"none", "auto"}:
            raise ValueError("metric_learning must be 'none' or 'auto'")
        self.metric_learning = metric


class ENNTS:
    def __init__(self, config: ENNTSConfig) -> None:
        cfg = config
        self._inner = _RustENNTS(
            int(cfg.lookback),
            int(cfg.num_tables),
            int(cfg.num_projections),
            float(cfg.bucket_width),
            int(cfg.k),
            int(cfg.seed),
            int(cfg.horizon),
            int(cfg.cadence),
            int(cfg.probe),
            (
                float(cfg.epistemic_variance_scale),
                float(cfg.aleatoric_variance_scale),
                str(cfg.metric_learning),
            ),
        )

    def add(
        self,
        series_id: int,
        x: np.ndarray,
        y: np.ndarray | None = None,
        timestamps: np.ndarray | None = None,
    ) -> None:
        x_arr = _feature_matrix(x)
        y_arr = None if y is None else _target_matrix(y)
        ts = None if timestamps is None else _timestamp_vector(timestamps)
        self._inner.add(int(series_id), x_arr, y_arr, ts)

    def add_y(self, series_id: int, timestamps: np.ndarray, y: np.ndarray) -> None:
        self._inner.add_y(
            int(series_id),
            _timestamp_vector(timestamps),
            _target_matrix(y),
        )

    def sync_index(self) -> None:
        self._inner.sync_index()

    def search(
        self,
        query: np.ndarray,
        cutoff: int,
        k: int | None = None,
        require_target: bool = True,
    ):
        flat = _flat_query(self._inner, query)
        return self._inner.search(flat, int(cutoff), k, bool(require_target))

    def search_exact(
        self,
        query: np.ndarray,
        cutoff: int,
        k: int | None = None,
        require_target: bool = True,
    ):
        flat = _flat_query(self._inner, query)
        return self._inner.search_exact(flat, int(cutoff), k, bool(require_target))

    def posterior(
        self,
        query: np.ndarray,
        cutoff: int,
        k: int | None = None,
        observation_noise: bool = True,
    ):
        flat = _flat_query(self._inner, query)
        return self._inner.posterior(flat, int(cutoff), k, bool(observation_noise))

    def set_variance_scales(self, epistemic: float, aleatoric: float) -> None:
        self._inner.set_variance_scales(float(epistemic), float(aleatoric))

    @property
    def num_obs(self) -> int:
        return int(self._inner.num_obs)

    @property
    def num_windows(self) -> int:
        return int(self._inner.num_windows)

    @property
    def num_dim(self) -> int:
        dim = self._inner.num_dim
        if dim is None:
            raise RuntimeError("ENNTS has no observations yet")
        return int(dim)

    @property
    def lookback(self) -> int:
        return int(self._inner.lookback)

    @property
    def metric_weights(self) -> np.ndarray | None:
        weights = self._inner.metric_weights
        if weights is None:
            return None
        return np.asarray(weights, dtype=float)

    @property
    def heldout_gain(self) -> float | None:
        gain = self._inner.heldout_gain
        if gain is None:
            return None
        return float(gain)

    @property
    def metric_morph_remaining(self) -> int:
        return int(self._inner.metric_morph_remaining)

    @property
    def uses_learned_metric(self) -> bool:
        return bool(self._inner.uses_learned_metric)


def _feature_matrix(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("x must have shape (n_obs, n_features)")
    return arr


def _target_matrix(y: np.ndarray) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("y must have shape (n_obs,) or (n_obs, n_targets)")
    return arr


def _timestamp_vector(timestamps: np.ndarray) -> np.ndarray:
    return np.asarray(timestamps, dtype=np.int64)


def _flat_query(inner, query: np.ndarray) -> np.ndarray:
    arr = np.asarray(query, dtype=np.float64)
    lookback = int(inner.lookback)
    dim = inner.num_dim
    if dim is None:
        raise ValueError("ENNTS has no observations yet")
    width = lookback * int(dim)
    if arr.ndim == 2 and arr.shape == (lookback, int(dim)):
        return arr.reshape(1, width)
    if arr.ndim == 3 and arr.shape[1:] == (lookback, int(dim)):
        return arr.reshape(arr.shape[0], width)
    if arr.ndim == 2 and arr.shape[1] == width:
        return np.ascontiguousarray(arr)
    raise ValueError(
        "query must have shape (lookback, n_features), "
        "(n_query, lookback, n_features), or (n_query, lookback * n_features)"
    )
