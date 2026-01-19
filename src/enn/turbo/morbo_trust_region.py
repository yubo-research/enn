from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator
    from scipy.stats._qmc import QMCEngine

    from .config.morbo_tr_config import MorboTRConfig
    from .config.rescalarize import Rescalarize


class MorboTrustRegion:
    def __init__(
        self,
        config: MorboTRConfig,
        num_dim: int,
        *,
        rng: Generator,
    ) -> None:
        from .components.incumbent_selector import ChebyshevIncumbentSelector
        from .config.turbo_tr_config import TurboTRConfig
        from .turbo_trust_region import TurboTrustRegion

        self._config = config
        inner_config = TurboTRConfig(length=config.length)
        self._tr = TurboTrustRegion(
            config=inner_config,
            num_dim=num_dim,
        )
        self._num_dim = int(num_dim)
        self._num_metrics = int(config.num_metrics)
        if self._num_metrics <= 0:
            raise ValueError(self._num_metrics)

        self._alpha = float(config.alpha)
        self._rescalarize = config.rescalarize
        self.incumbent_selector = ChebyshevIncumbentSelector(
            num_metrics=self._num_metrics,
            alpha=self._alpha,
            noise_aware=config.noise_aware,
        )
        self.incumbent_selector.reset(rng)
        self._weights = self.incumbent_selector.weights

        self._y_min: np.ndarray | Any | None = None
        self._y_max: np.ndarray | Any | None = None

    @property
    def num_dim(self) -> int:
        return self._num_dim

    @property
    def num_metrics(self) -> int:
        return self._num_metrics

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    @property
    def length(self) -> float:
        return float(self._tr.length)

    @property
    def rescalarize(self) -> Rescalarize:
        return self._rescalarize

    def resample_weights(self, rng: Generator) -> None:
        self.incumbent_selector.reset(rng)
        self._weights = self.incumbent_selector.weights

    def _update_ranges(self, y_obs, prev_n):
        y_min_all, y_max_all = y_obs.min(axis=0), y_obs.max(axis=0)
        y_min_prev = y_obs[:prev_n].min(axis=0) if prev_n > 0 else y_min_all
        y_max_prev = y_obs[:prev_n].max(axis=0) if prev_n > 0 else y_max_all
        self._y_min, self._y_max = y_min_all, y_max_all
        return y_min_prev, y_max_prev

    def update(self, y_obs: np.ndarray | Any) -> None:
        import numpy as np

        y_obs = np.asarray(y_obs, dtype=float)
        if y_obs.ndim != 2 or y_obs.shape[1] != self._num_metrics:
            raise ValueError((y_obs.shape, self._num_metrics))
        n = int(y_obs.shape[0])
        if n == 0:
            self._y_min, self._y_max = None, None
            self._tr.restart()
            return
        prev_n = int(self._tr.prev_num_obs)
        if n < prev_n:
            raise ValueError((n, prev_n))
        y_min_prev, y_max_prev = self._update_ranges(y_obs, prev_n)
        if prev_n == 0:
            values = np.asarray(self.scalarize(y_obs, clip=True), dtype=float)
            if values.shape != (n,):
                raise RuntimeError((values.shape, n))
            self._tr.update(values)
            return
        if not np.isfinite(self._tr.best_value):
            raise RuntimeError(self._tr.best_value)
        values_old = np.asarray(
            self._scalarize_with_ranges(
                y_obs, y_min=y_min_prev, y_max=y_max_prev, clip=True
            ),
            dtype=float,
        )
        if values_old.shape != (n,):
            raise RuntimeError((values_old.shape, n))
        self._tr.best_value = float(np.max(values_old[:prev_n]))
        if prev_n < n:
            self._tr.update(values_old)

    def scalarize(self, y: np.ndarray | Any, *, clip: bool) -> np.ndarray:
        import numpy as np

        y = np.asarray(y, dtype=float)
        if y.ndim != 2 or y.shape[1] != self._num_metrics:
            raise ValueError(y.shape)
        if self._y_min is None or self._y_max is None:
            raise RuntimeError("scalarize called before any observations")

        return self._scalarize_with_ranges(
            y, y_min=self._y_min, y_max=self._y_max, clip=clip
        )

    def _scalarize_with_ranges(
        self,
        y: np.ndarray | Any,
        *,
        y_min: np.ndarray,
        y_max: np.ndarray,
        clip: bool,
    ) -> np.ndarray:
        import numpy as np

        y = np.asarray(y, dtype=float)
        if y.ndim != 2 or y.shape[1] != self._num_metrics:
            raise ValueError(y.shape)
        y_min = np.asarray(y_min, dtype=float).reshape(-1)
        y_max = np.asarray(y_max, dtype=float).reshape(-1)
        if y_min.shape != (self._num_metrics,) or y_max.shape != (self._num_metrics,):
            raise ValueError((y_min.shape, y_max.shape, self._num_metrics))

        denom = y_max - y_min
        is_deg = denom <= 0.0
        denom_safe = np.where(is_deg, 1.0, denom)
        z = (y - y_min.reshape(1, -1)) / denom_safe.reshape(1, -1)
        z = np.where(is_deg, 0.5, z)
        if clip:
            z = np.clip(z, 0.0, 1.0)
        t = z * self._weights.reshape(1, -1)
        scores = np.min(t, axis=1) + self._alpha * np.sum(t, axis=1)
        return scores

    def needs_restart(self) -> bool:
        return self._tr.needs_restart()

    def restart(self, rng: Generator | None = None) -> None:
        from .config.rescalarize import Rescalarize

        self._y_min = None
        self._y_max = None
        self._tr.restart()
        if rng is not None and self._rescalarize == Rescalarize.ON_RESTART:
            self.resample_weights(rng)

    def validate_request(self, num_arms: int, *, is_fallback: bool = False) -> None:
        return self._tr.validate_request(num_arms, is_fallback=is_fallback)

    def compute_bounds_1d(
        self, x_center: np.ndarray | Any, lengthscales: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._tr.compute_bounds_1d(x_center, lengthscales)

    def generate_candidates(
        self,
        x_center: np.ndarray,
        lengthscales: np.ndarray | None,
        num_candidates: int,
        rng: Generator,
        sobol_engine: QMCEngine,
    ) -> np.ndarray:
        from .tr_helpers import generate_tr_candidates

        return generate_tr_candidates(
            self._tr.compute_bounds_1d,
            x_center,
            lengthscales,
            num_candidates,
            rng=rng,
            sobol_engine=sobol_engine,
        )

    def get_incumbent_indices(
        self,
        y: np.ndarray | Any,
        rng: Generator,  # noqa: ARG002
    ) -> np.ndarray:
        import numpy as np

        y = np.asarray(y, dtype=float)
        if y.ndim != 2:
            raise ValueError(y.shape)
        n = y.shape[0]
        if n == 0:
            return np.array([], dtype=int)

        from nds import ndomsort

        idx_front = np.array(ndomsort.non_domin_sort(-y, only_front_indices=True))
        return np.where(idx_front == 0)[0]
