from __future__ import annotations
from typing import TYPE_CHECKING, Any
import numpy as np
from .posterior_result import PosteriorResult
from .surrogate_result import SurrogateResult

if TYPE_CHECKING:
    from numpy.random import Generator
    from ..config.surrogate import ENNSurrogateConfig


class ENNSurrogate:
    def __init__(self, config: ENNSurrogateConfig) -> None:
        self._config = config
        self._enn: Any | None = None
        self._params: Any | None = None

    @property
    def lengthscales(self) -> np.ndarray | None:
        return None

    def fit(
        self,
        x_obs: np.ndarray,
        y_obs: np.ndarray,
        y_var: np.ndarray | None = None,
        *,
        num_steps: int = 0,
        rng: Generator | None = None,
    ) -> SurrogateResult:
        from ..proposal import mk_enn
        from ..config.enums import ENNIndexDriver

        k = self._config.k if self._config.k is not None else 10
        if (
            self._config.index_driver == ENNIndexDriver.HNSW
            and self._enn is not None
            and len(x_obs) > len(self._enn)
        ):
            n_old = len(self._enn)
            new_x = x_obs[n_old:]
            new_y = y_obs[n_old:]
            new_yvar = y_var[n_old:] if y_var is not None else None
            self._enn.add(new_x, new_y, new_yvar)
            if self._config.num_fit_samples is not None and rng is not None:
                from ...enn.enn_fit import enn_fit

                self._params = enn_fit(
                    self._enn,
                    k=k,
                    num_fit_candidates=self._config.num_fit_candidates
                    if self._config.num_fit_candidates is not None
                    else 30,
                    num_fit_samples=self._config.num_fit_samples,
                    rng=rng,
                    params_warm_start=self._params,
                )
        else:
            self._enn, self._params = mk_enn(
                list(x_obs),
                list(y_obs),
                k,
                list(y_var) if y_var is not None else [],
                num_fit_samples=self._config.num_fit_samples,
                num_fit_candidates=self._config.num_fit_candidates,
                scale_x=self._config.scale_x,
                index_driver=self._config.index_driver,
                rng=rng,
                params_warm_start=self._params,
            )
        return SurrogateResult(model=self._enn, lengthscales=None)

    def get_incumbent_candidate_indices(self, y_obs: np.ndarray) -> np.ndarray:
        y_array = np.asarray(y_obs, dtype=float)
        k = self._config.k
        if k is None:
            num_fit_candidates = (
                self._config.num_fit_candidates
                if self._config.num_fit_candidates is not None
                else 100
            )
            k = min(len(y_array), max(10, 2 * num_fit_candidates))
        if y_array.ndim == 2 and y_array.shape[1] > 1:
            num_top = min(k, len(y_array))
            union_indices: set[int] = set()
            for m in range(y_array.shape[1]):
                top_m = np.argpartition(-y_array[:, m], num_top - 1)[:num_top]
                union_indices.update(top_m.tolist())
            return np.array(sorted(union_indices), dtype=int)
        else:
            y_flat = y_array[:, 0] if y_array.ndim == 2 else y_array
            num_top = min(k, len(y_flat))
            return np.argpartition(-y_flat, num_top - 1)[:num_top]

    def predict(self, x: np.ndarray) -> PosteriorResult:
        if self._enn is None or self._params is None:
            raise RuntimeError("ENNSurrogate.predict requires a fitted model")
        posterior = self._enn.posterior(x, params=self._params)
        return PosteriorResult(mu=posterior.mu, sigma=posterior.se)

    def sample(self, x: np.ndarray, num_samples: int, rng: Generator) -> np.ndarray:
        if self._enn is None or self._params is None:
            raise RuntimeError("ENNSurrogate.sample requires a fitted model")
        num_candidates = len(x)
        num_metrics = self._enn.num_outputs
        base_seed = rng.integers(0, 2**31)
        function_seeds = np.arange(base_seed, base_seed + num_samples, dtype=np.int64)
        samples = self._enn.posterior_function_draw(
            x, self._params, function_seeds=function_seeds
        )
        assert samples.shape == (num_samples, num_candidates, num_metrics), (
            f"ENN samples shape mismatch: got {samples.shape}, "
            f"expected ({num_samples}, {num_candidates}, {num_metrics})"
        )
        return samples
