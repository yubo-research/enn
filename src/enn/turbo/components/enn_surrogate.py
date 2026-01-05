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

    def fit(
        self,
        x_obs: np.ndarray,
        y_obs: np.ndarray,
        y_var: np.ndarray | None = None,
        *,
        num_steps: int = 0,  # noqa: ARG002
        rng: Generator | None = None,
    ) -> SurrogateResult:
        from ..proposal import mk_enn

        k = self._config.k if self._config.k is not None else 10

        self._enn, self._params = mk_enn(
            list(x_obs),
            list(y_obs),
            k,
            list(y_var) if y_var is not None else [],
            num_fit_samples=self._config.num_fit_samples,
            num_fit_candidates=self._config.num_fit_candidates,
            scale_x=self._config.scale_x,
            rng=rng,
            params_warm_start=self._params,
        )

        return SurrogateResult(model=self._enn, lengthscales=None)

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

        # ENN output: (num_samples, num_candidates, num_metrics)
        assert samples.shape == (num_samples, num_candidates, num_metrics), (
            f"ENN samples shape mismatch: got {samples.shape}, "
            f"expected ({num_samples}, {num_candidates}, {num_metrics})"
        )
        return samples
