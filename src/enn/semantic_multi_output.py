"""Optional semantic output sharing with row-wise out-of-fold calibration."""

from dataclasses import dataclass

import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams


@dataclass(frozen=True)
class OutputPosterior:
    """Marginal Gaussian predictions in original units; no joint covariance."""

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
        self.independent = EpistemicNearestNeighbors(x, z)
        self.latent = (
            EpistemicNearestNeighbors(x, z @ loadings) if loadings.shape[1] else None
        )

    def predict(self, x, alpha):
        """Use a triangle-inequality SD envelope for correlated component errors."""
        posterior = self.independent.posterior(x, params=self.params)
        independent, independent_se = posterior.mu, posterior.se
        if self.latent and alpha > 0:
            posterior = self.latent.posterior(x, params=self.params)
            latent, latent_se = posterior.mu, posterior.se
            shared = latent @ self.loadings.T
            shared_se = latent_se @ np.abs(self.loadings.T)
            independent = (1 - alpha) * independent + alpha * shared
            independent_se = (1 - alpha) * independent_se + alpha * shared_se
        independent[:, self.constant] = 0
        independent_se[:, self.constant] = 0
        return independent * self.scale + self.mean, independent_se * self.scale


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
    """Blend independent and semantic latent ENNs with fixed hyperparameters.

    ``projection`` is a SemanticOutputProjection built solely from metadata.
    ``fit`` refits all scaling and ENNs within each row fold, calibrates marginal
    variances, then trains on all supplied rows. Select rank, alpha, and params
    on separate validation data or nested CV. Do not change them after fitting.
    At alpha=0 means reproduce independent ENNs (uncertainties are calibrated).
    Scalar output and latent predictions are batched through the existing ENN
    multi-output API to reuse neighbor searches. Constant training outputs remain
    at their training mean.
    """

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
        """Refit row folds and estimate a per-output Gaussian likelihood scale."""
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
