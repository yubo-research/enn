from __future__ import annotations

from typing import Any

import numpy as np

from enn._rust import ENNParams as RustENNParams
from enn._rust import ENNStatefulFitter as _RustENNStatefulFitter

from .affine_calibrator import AffineCalibrator


class ENNStatefulFitter:
    def __init__(
        self,
        k: int,
        rng: Any,
        *,
        infer_aleatoric_variance_scale: bool = True,
        affine_calibrate: bool = False,
    ) -> None:
        seed = int(rng.integers(0, 2**63 - 1))
        self._rng = np.random.default_rng(seed)
        self._rust = _RustENNStatefulFitter(
            k,
            seed,
            infer_aleatoric_variance_scale,
        )
        self._affine_calibrate = bool(affine_calibrate)
        self._affine_calibrator: AffineCalibrator | None = None

    @property
    def affine_calibrator(self) -> AffineCalibrator | None:
        """Fitted calibrator when ``affine_calibrate=True``; else None."""
        return self._affine_calibrator

    def tell(
        self,
        x: np.ndarray,
        y: np.ndarray,
        yvar: np.ndarray | None = None,
    ) -> None:
        """Register a batch for incremental y_std; must match rows added to the model."""
        x_array = np.asarray(x, dtype=float)
        y_array = np.asarray(y, dtype=float)
        if y_array.ndim == 1:
            y_array = y_array.reshape(-1, 1)
        yvar_array = None
        if yvar is not None:
            yvar_array = np.asarray(yvar, dtype=float)
            if yvar_array.ndim == 1:
                yvar_array = yvar_array.reshape(-1, 1)
        self._rust.tell(x_array, y_array, yvar_array)

    def y_std(self) -> np.ndarray:
        return np.asarray(self._rust.y_std(), dtype=float)

    def ask(
        self,
        model: Any,
        *,
        num_fit_candidates: int,
        num_fit_samples: int,
        params_warm_start: Any | None = None,
    ) -> Any:
        """Fit hyperparameters; tell row count must equal model.num_obs() or y_std is wrong."""
        from .enn_class import EpistemicNearestNeighbors as PyENN
        from .enn_params import ENNParams as PyENNParams

        if not isinstance(model, PyENN):
            raise TypeError(f"Expected EpistemicNearestNeighbors, got {type(model)}")

        rust_warm_start = None
        if params_warm_start is not None:
            rust_warm_start = RustENNParams(
                params_warm_start.k_num_neighbors,
                params_warm_start.epistemic_variance_scale,
                params_warm_start.aleatoric_variance_scale,
            )

        rust_result = self._rust.ask(
            model.rust_backend,
            num_fit_candidates,
            num_fit_samples,
            rust_warm_start,
        )

        result = PyENNParams(
            k_num_neighbors=rust_result.k_num_neighbors,
            epistemic_variance_scale=rust_result.epistemic_variance_scale,
            aleatoric_variance_scale=rust_result.aleatoric_variance_scale,
        )
        if self._affine_calibrate:
            self._affine_calibrator = _fit_loo_affine(
                model, result, num_fit_samples, self._rng
            )
        else:
            self._affine_calibrator = None
        return result


def _fit_loo_affine(
    model: Any,
    params: Any,
    num_fit_samples: int,
    rng: np.random.Generator,
) -> AffineCalibrator:
    from .enn_params import PosteriorFlags

    n = len(model)
    m = int(model.num_outputs)
    if n < 2:
        return AffineCalibrator.identity(m)
    p = min(int(num_fit_samples), n)
    indices = rng.choice(n, size=p, replace=False)
    x_loo, y_loo, _ = model.train_rows_at(indices)
    flags = PosteriorFlags(exclude_nearest=True, observation_noise=True)
    post = model.posterior(x_loo, params=params, flags=flags)
    cal = AffineCalibrator.identity(m)
    cal.fit(post.mu, y_loo)
    cal.fit_residual_scale(post.mu, post.se, y_loo)
    return cal
