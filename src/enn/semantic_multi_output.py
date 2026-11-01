from dataclasses import dataclass

import numpy as np

from enn._rust import SemanticMultiOutputENN as _NativeMultiOutput
from enn.enn.enn_params import ENNParams


@dataclass(frozen=True)
class OutputPosterior:
    mu: np.ndarray
    se: np.ndarray


class SemanticMultiOutputENN:
    def __init__(self, projection, *, alpha=0.3, params=None, folds=5, seed=0):
        params = params if params is not None else ENNParams(8, 1.0, 0.0)
        self._projection = projection
        self._alpha = float(alpha)
        self._native = _NativeMultiOutput(
            projection.loadings,
            self._alpha,
            params.k_num_neighbors,
            params.epistemic_variance_scale,
            params.aleatoric_variance_scale,
            folds,
            seed,
        )

    @property
    def projection(self):
        return self._projection

    @property
    def alpha(self):
        return self._alpha

    @property
    def oof_mean_(self):
        return self._native.oof_mean

    @property
    def oof_variance_(self):
        return self._native.oof_variance

    @property
    def fold_ids_(self):
        return self._native.fold_ids

    def fit(self, x, y):
        self._native.fit(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        return self

    def posterior(self, x):
        return OutputPosterior(*self._native.posterior(np.asarray(x, dtype=float)))
