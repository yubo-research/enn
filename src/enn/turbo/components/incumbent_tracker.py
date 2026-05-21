from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..config.surrogate import SurrogateConfig

_ALL_CANDIDATES_M_THRESHOLD = 10**8


def incumbent_tracker_m_from_surrogate_config(surrogate_cfg: SurrogateConfig) -> int:
    if type(surrogate_cfg).__name__ != "ENNSurrogateConfig":
        return 10**9
    k = surrogate_cfg.k
    if k is not None:
        return int(k)
    num_fit_candidates = (
        surrogate_cfg.num_fit_candidates
        if surrogate_cfg.num_fit_candidates is not None
        else 100
    )
    return max(10, 2 * num_fit_candidates)


def _rank_key(index: int, value: float) -> tuple[float, int]:
    return (-float(value), int(index))


class _ScalarTopM:
    def __init__(self, m: int) -> None:
        self._m = int(m)
        self._entries: list[tuple[int, float]] = []

    def tell(self, index: int, value: float) -> None:
        self._entries.append((int(index), float(value)))
        self._entries.sort(key=lambda e: _rank_key(e[0], e[1]))
        del self._entries[self._m :]

    def ask(self) -> np.ndarray:
        if not self._entries:
            return np.array([], dtype=int)
        return np.array(sorted(i for i, _ in self._entries), dtype=int)

    def reset(self) -> None:
        self._entries.clear()


class _NoiselessMax:
    def __init__(self) -> None:
        self._max_y = -np.inf
        self._indices: list[int] = []

    def tell(self, index: int, value: float) -> None:
        y = float(value)
        if y > self._max_y:
            self._max_y = y
            self._indices = [int(index)]
        elif y == self._max_y:
            self._indices.append(int(index))

    def ask(self) -> np.ndarray:
        return np.array(sorted(self._indices), dtype=int)

    def reset(self) -> None:
        self._max_y = -np.inf
        self._indices.clear()


class IncrementalIncumbentTracker:
    def __init__(self, *, m: int, noise_aware: bool, num_metrics: int = 1) -> None:
        if m < 1:
            raise ValueError(f"m must be >= 1, got {m}")
        if num_metrics < 1:
            raise ValueError(f"num_metrics must be >= 1, got {num_metrics}")
        self._m = int(m)
        self._noise_aware = bool(noise_aware)
        self._num_metrics = int(num_metrics)
        self._observation_count = 0
        self._all_candidates = m >= _ALL_CANDIDATES_M_THRESHOLD
        self._all_indices: list[int] = []
        self._max_state: _NoiselessMax | None = None
        self._topm: _ScalarTopM | None = None
        self._per_metric: list[_ScalarTopM] | None = None
        if self._all_candidates:
            return
        if self._num_metrics == 1 and not self._noise_aware:
            self._max_state = _NoiselessMax()
        elif self._num_metrics == 1:
            self._topm = _ScalarTopM(self._m)
        else:
            self._per_metric = [_ScalarTopM(self._m) for _ in range(self._num_metrics)]

    def tell(self, index: int, y: float | np.ndarray) -> None:
        row = self._coerce_y_row(y)
        self._observation_count += 1
        idx = int(index)
        if self._all_candidates:
            self._all_indices.append(idx)
            return
        if self._max_state is not None:
            self._max_state.tell(idx, row[0])
            return
        if self._topm is not None:
            self._topm.tell(idx, row[0])
            return
        assert self._per_metric is not None
        for col, state in enumerate(self._per_metric):
            state.tell(idx, row[col])

    def ask(self) -> np.ndarray:
        if self._observation_count == 0:
            return np.array([], dtype=int)
        if self._all_candidates:
            return np.array(sorted(self._all_indices), dtype=int)
        if self._max_state is not None:
            return self._max_state.ask()
        if self._topm is not None:
            return self._topm.ask()
        assert self._per_metric is not None
        union: set[int] = set()
        for state in self._per_metric:
            union.update(state.ask().tolist())
        return np.array(sorted(union), dtype=int)

    def observation_count(self) -> int:
        return self._observation_count

    def reset(self) -> None:
        self._observation_count = 0
        self._all_indices.clear()
        if self._max_state is not None:
            self._max_state.reset()
        if self._topm is not None:
            self._topm.reset()
        if self._per_metric is not None:
            for state in self._per_metric:
                state.reset()

    def rebuild(self, y_obs: np.ndarray) -> None:
        self.reset()
        y_array = np.asarray(y_obs, dtype=float)
        if y_array.size == 0:
            return
        if y_array.ndim == 1:
            for i, y_val in enumerate(y_array):
                self.tell(i, float(y_val))
            return
        if y_array.ndim != 2 or y_array.shape[1] != self._num_metrics:
            raise ValueError((y_array.shape, self._num_metrics))
        for i in range(y_array.shape[0]):
            self.tell(i, y_array[i])

    def _coerce_y_row(self, y: float | np.ndarray) -> np.ndarray:
        if self._num_metrics == 1:
            if isinstance(y, (float, int, np.floating, np.integer)):
                return np.array([float(y)], dtype=float)
            arr = np.asarray(y, dtype=float).reshape(-1)
            if arr.size != 1:
                raise ValueError((arr.shape, self._num_metrics))
            return arr
        arr = np.asarray(y, dtype=float).reshape(-1)
        if arr.size != self._num_metrics:
            raise ValueError((arr.shape, self._num_metrics))
        return arr


def build_incumbent_tracker(
    surrogate_cfg: SurrogateConfig,
    tr_state: Any,
) -> IncrementalIncumbentTracker:
    m = incumbent_tracker_m_from_surrogate_config(surrogate_cfg)
    num_metrics = int(getattr(tr_state, "num_metrics", 1))
    noise_aware = False
    if hasattr(tr_state, "config"):
        noise_aware = bool(getattr(tr_state.config, "noise_aware", False))
    if hasattr(tr_state, "incumbent_selector"):
        noise_aware = noise_aware or bool(
            getattr(tr_state.incumbent_selector, "noise_aware", False)
        )
    return IncrementalIncumbentTracker(
        m=m, noise_aware=noise_aware, num_metrics=num_metrics
    )
