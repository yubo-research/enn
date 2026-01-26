from __future__ import annotations

from typing import TYPE_CHECKING, Any
import numpy as np

from .turbo_utils import ScalarIncumbentMixin

if TYPE_CHECKING:
    from numpy.random import Generator
    from scipy.stats._qmc import QMCEngine

    from .config.morbo_tr_config import MorboTRConfig
    from .config.rescalarize import Rescalarize

from .config.candidate_rv import CandidateRV
from .config.raasp_driver import RAASPDriver


class MorboTrustRegion(ScalarIncumbentMixin):
    def __init__(
        self,
        config: MorboTRConfig,
        num_dim: int,
        *,
        rng: Generator,
        candidate_rv: CandidateRV = CandidateRV.SOBOL,
    ) -> None:
        from .components.incumbent_selector import ChebyshevIncumbentSelector
        from .config.turbo_tr_config import TurboTRConfig
        from .turbo_trust_region import TurboTrustRegion

        self._config = config
        self._candidate_rv = candidate_rv
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
        self._incumbent_y_raw: np.ndarray | None = None

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

    def _update_ranges(self, y_obs):
        self._y_min, self._y_max = y_obs.min(axis=0), y_obs.max(axis=0)
        assert self._y_min.shape == (self._num_metrics,) and self._y_max.shape == (
            self._num_metrics,
        )

    def update(self, y_obs: np.ndarray, y_incumbent: np.ndarray) -> None:
        y_obs = np.asarray(y_obs, dtype=float)
        if y_obs.ndim != 2 or y_obs.shape[1] != self._num_metrics:
            raise ValueError((y_obs.shape, self._num_metrics))
        n = int(y_obs.shape[0])
        if n == 0:
            self._y_min, self._y_max, self._incumbent_y_raw = None, None, None
            self._tr.restart()
            return
        prev_n = int(self._tr.prev_num_obs)
        if n < prev_n:
            raise ValueError((n, prev_n))
        self._y_min, self._y_max = y_obs.min(axis=0), y_obs.max(axis=0)
        assert self._y_min.shape == (self._num_metrics,) and self._y_max.shape == (
            self._num_metrics,
        )
        y_incumbent = np.asarray(y_incumbent, dtype=float).reshape(1, -1)
        if y_incumbent.shape != (1, self._num_metrics):
            raise ValueError(
                f"y_incumbent must have shape (1, {self._num_metrics}), got {y_incumbent.shape}"
            )
        if prev_n == 0 or self._incumbent_y_raw is None:
            self._handle_initial_update(y_incumbent, n)
            return
        scores = self.scalarize(
            np.vstack([self._incumbent_y_raw, y_incumbent]), clip=True
        )
        old_score, new_score = float(scores[0]), float(scores[1])
        self._tr.best_value = old_score
        self._tr.update(np.zeros((n, 1)), np.array([new_score]))
        if new_score > old_score:
            self._incumbent_y_raw = y_incumbent.copy()

    def _handle_initial_update(self, y_incumbent: np.ndarray, n: int) -> None:
        self._incumbent_y_raw = y_incumbent.copy()
        score = self.scalarize(y_incumbent, clip=True)
        dummy_y_obs = np.zeros((n, 1))
        self._tr.update(dummy_y_obs, score)

    def scalarize(self, y: np.ndarray, *, clip: bool) -> np.ndarray:
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
        y: np.ndarray,
        *,
        y_min: np.ndarray,
        y_max: np.ndarray,
        clip: bool,
    ) -> np.ndarray:
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

        self._y_min, self._y_max, self._incumbent_y_raw = None, None, None
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
        raasp_driver: RAASPDriver = RAASPDriver.ORIG,
        num_pert: int = 20,
    ) -> np.ndarray:
        from .turbo_utils import generate_tr_candidates

        return generate_tr_candidates(
            self._tr.compute_bounds_1d,
            x_center,
            lengthscales,
            num_candidates,
            rng=rng,
            candidate_rv=self._candidate_rv,
            sobol_engine=sobol_engine,
            raasp_driver=raasp_driver,
            num_pert=num_pert,
        )

    def get_incumbent_indices(
        self,
        y: np.ndarray | Any,
        rng: Generator,
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

    def get_incumbent_value(
        self,
        y_obs: np.ndarray | Any,
        rng: Generator,
        mu_obs: np.ndarray | None = None,
    ) -> np.ndarray:
        import numpy as np

        y_obs = np.asarray(y_obs, dtype=float)
        if y_obs.ndim != 2 or y_obs.shape[1] != self._num_metrics:
            raise ValueError((y_obs.shape, self._num_metrics))
        n = int(y_obs.shape[0])
        if n == 0:
            return np.array([], dtype=float)
        idx = self.get_incumbent_index(y_obs, rng, mu=mu_obs)
        use_mu = bool(getattr(self.incumbent_selector, "noise_aware", False))
        values = np.asarray(mu_obs if use_mu else y_obs, dtype=float)
        if values.ndim != 2 or values.shape[1] != self._num_metrics:
            raise ValueError((values.shape, self._num_metrics))
        return values[idx : idx + 1].copy()
