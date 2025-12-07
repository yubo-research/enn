from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator


@dataclass(frozen=True)
class CEMConfig:
    num_samples: int = 100
    elite_frac: float = 0.1
    init_std: float = 0.5
    min_std: float = 1e-4
    smoothing: float = 0.0
    fixed_center: bool = False


class CEMSampler:
    def __init__(
        self,
        num_dim: int,
        *,
        lb: np.ndarray | Any | None = None,
        ub: np.ndarray | Any | None = None,
        x_center: np.ndarray | Any | None = None,
        config: CEMConfig | None = None,
        rng: Generator | Any,
    ) -> None:
        import numpy as np

        if config is None:
            config = CEMConfig()
        self._config = config
        self._num_dim = num_dim
        self._rng = rng

        if lb is None:
            self._lb = np.zeros(num_dim, dtype=float)
        else:
            self._lb = np.asarray(lb, dtype=float)

        if ub is None:
            self._ub = np.ones(num_dim, dtype=float)
        else:
            self._ub = np.asarray(ub, dtype=float)

        if self._lb.shape != (num_dim,) or self._ub.shape != (num_dim,):
            raise ValueError(
                f"Bounds shape mismatch: lb={self._lb.shape}, ub={self._ub.shape}, num_dim={num_dim}"
            )
        if np.any(self._lb >= self._ub):
            raise ValueError("Lower bounds must be strictly less than upper bounds")

        if x_center is None:
            x_center = self._lb + self._rng.random(num_dim) * (self._ub - self._lb)
        else:
            x_center = np.asarray(x_center, dtype=float)

        self._mu = np.clip(x_center, self._lb, self._ub)
        self._std = np.full(
            num_dim, config.init_std * (self._ub - self._lb), dtype=float
        )

        self._num_elite = max(1, int(config.num_samples * config.elite_frac))
        self._best_x: np.ndarray = self._mu.copy()
        self._best_score: float = float("-inf")
        self._iteration: int = 0
        self._pending_samples: np.ndarray | None = None

    @property
    def best_x(self) -> np.ndarray:
        return self._best_x.copy()

    @property
    def best_score(self) -> float:
        return self._best_score

    @property
    def mu(self) -> np.ndarray:
        return self._mu.copy()

    @property
    def std(self) -> np.ndarray:
        return self._std.copy()

    @property
    def iteration(self) -> int:
        return self._iteration

    def ask(self) -> np.ndarray:
        samples = _sample_truncated_normal(
            self._mu,
            self._std,
            self._lb,
            self._ub,
            self._config.num_samples,
            self._rng,
        )
        self._pending_samples = samples
        return samples

    def tell(self, scores: np.ndarray | Any) -> None:
        import numpy as np

        scores = np.asarray(scores, dtype=float)
        if self._pending_samples is None:
            raise RuntimeError("Must call ask() before tell()")
        if scores.shape != (self._config.num_samples,):
            raise ValueError(
                f"scores shape {scores.shape} does not match num_samples ({self._config.num_samples},)"
            )

        samples = self._pending_samples
        self._pending_samples = None

        elite_idx = np.argpartition(-scores, self._num_elite - 1)[: self._num_elite]
        elite_samples = samples[elite_idx]
        elite_scores = scores[elite_idx]

        iter_best_idx = np.argmax(elite_scores)
        iter_best_score = elite_scores[iter_best_idx]
        if iter_best_score > self._best_score:
            self._best_score = float(iter_best_score)
            self._best_x = elite_samples[iter_best_idx].copy()

        if self._config.fixed_center:
            new_std = np.sqrt(np.mean((elite_samples - self._mu) ** 2, axis=0))
        else:
            new_mu = np.mean(elite_samples, axis=0)
            new_std = np.std(elite_samples, axis=0)

        new_std = np.maximum(new_std, self._config.min_std)

        if self._config.smoothing > 0:
            self._std = (
                self._config.smoothing * self._std
                + (1 - self._config.smoothing) * new_std
            )
            if not self._config.fixed_center:
                self._mu = (
                    self._config.smoothing * self._mu
                    + (1 - self._config.smoothing) * new_mu
                )
        else:
            self._std = new_std
            if not self._config.fixed_center:
                self._mu = new_mu

        self._iteration += 1


def _sample_truncated_normal(
    mu: np.ndarray | Any,
    std: np.ndarray | Any,
    lb: np.ndarray | Any,
    ub: np.ndarray | Any,
    num_samples: int,
    rng: Generator | Any,
) -> np.ndarray:
    import numpy as np
    from scipy.stats import truncnorm

    num_dim = mu.shape[-1]
    a = (lb - mu) / std
    b = (ub - mu) / std

    samples = truncnorm.rvs(
        a, b, loc=mu, scale=std, size=(num_samples, num_dim), random_state=rng
    )
    return np.asarray(samples, dtype=float)
