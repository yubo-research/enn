from dataclasses import dataclass

import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams


@dataclass(frozen=True)
class OutputPosterior:
    mu: np.ndarray
    se: np.ndarray


class _OutputModels:
    def __init__(self, x, y, loadings, params):
        self.mean = y.mean(axis=0)
        self.constant = np.ptp(y, axis=0) == 0
        self.scale = y.std(axis=0)
        self.scale[self.constant] = 1
        z = (y - self.mean) / self.scale
        self.loadings = loadings
        self.params = ENNParams(
            min(params.k_num_neighbors, len(x)),
            params.epistemic_variance_scale,
            params.aleatoric_variance_scale,
        )
        self.model = EpistemicNearestNeighbors(x, z)
        self.latent_scale = _latent_scale(z @ loadings) if loadings.shape[1] else None

    def predict(self, x, alpha):
        posterior = self.model.posterior(x, params=self.params)
        mean, se = posterior.mu, posterior.se
        if self.latent_scale is not None and alpha > 0:
            factor = se[:, :1] / self.model.rust_backend.y_scale_row[:, :1]
            shared_se = (factor * self.latent_scale) @ np.abs(self.loadings.T)
            shared = (mean @ self.loadings) @ self.loadings.T
            mean = (1 - alpha) * mean + alpha * shared
            se = (1 - alpha) * se + alpha * shared_se
        mean[:, self.constant] = 0
        se[:, self.constant] = 0
        return mean * self.scale + self.mean, se * self.scale


def _latent_scale(h):
    mean = h.mean(axis=0)
    std = np.sqrt(np.maximum((h**2).mean(axis=0) - mean**2, 0.0))
    return np.where(np.isfinite(std) & (std > 0), std, 1.0)


def _training_rows(x, y, outputs):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if (
        x.ndim != 2
        or x.shape[1] == 0
        or y.shape != (len(x), outputs)
        or len(x) < 4
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError(
            "training data must be finite matching matrices with at least four rows"
        )
    return x, y


class SemanticMultiOutputENN:
    def __init__(self, projection, *, alpha=0.3, params=None, folds=5, seed=0):
        if not np.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("alpha must be finite and in [0, 1]")
        if (
            isinstance(folds, bool)
            or not isinstance(folds, (int, np.integer))
            or folds < 2
        ):
            raise ValueError("folds must be an integer at least two")
        self._projection = projection
        self._alpha = float(alpha)
        self._params = params if params is not None else ENNParams(8, 1.0, 0.0)
        self._folds = int(folds)
        self._seed = seed
        self._fitted = None

    @property
    def projection(self):
        return self._projection

    @property
    def alpha(self):
        return self._alpha

    @property
    def _active_loadings(self):
        return (
            self.projection.loadings
            if self.alpha > 0
            else self.projection.loadings[:, :0]
        )

    def fit(self, x, y):
        x, y = _training_rows(x, y, len(self.projection.names))
        if self._folds > len(x):
            raise ValueError("fold count cannot exceed training row count")
        labels = self._fold_ids(len(x))
        oof_mu, oof_se = self._out_of_fold(x, y, labels)
        floor = np.maximum(y.std(axis=0) * 1e-6, 1e-12)
        raw_variance = np.maximum(oof_se**2, floor**2)
        calibration = np.maximum(
            np.mean((y - oof_mu) ** 2 / raw_variance, axis=0), 1e-12
        )
        fitted = _OutputModels(x, y, self._active_loadings, self._params)
        self._fitted = fitted
        self._dimension = x.shape[1]
        self._floor = floor
        self._calibration = calibration
        self.oof_mean_ = oof_mu
        self.oof_variance_ = raw_variance * calibration
        self.fold_ids_ = labels
        return self

    def _fold_ids(self, count):
        permutation = np.random.default_rng(self._seed).permutation(count)
        labels = np.empty(count, dtype=int)
        labels[permutation] = np.arange(count) % self._folds
        return labels

    def _out_of_fold(self, x, y, labels):
        mu, se = np.empty_like(y), np.empty_like(y)
        for fold in range(self._folds):
            held = labels == fold
            model = _OutputModels(
                x[~held], y[~held], self._active_loadings, self._params
            )
            mu[held], se[held] = model.predict(x[held], self.alpha)
        return mu, se

    def posterior(self, x):
        if self._fitted is None:
            raise RuntimeError("fit must be called before posterior")
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self._dimension or not np.isfinite(x).all():
            raise ValueError(
                "query must be a finite matrix with the fitted input dimension"
            )
        if not len(x):
            shape = (0, len(self.projection.names))
            return OutputPosterior(np.empty(shape), np.empty(shape))
        mu, se = self._fitted.predict(x, self.alpha)
        variance = np.maximum(se**2, self._floor**2) * self._calibration
        return OutputPosterior(mu, np.sqrt(variance))
