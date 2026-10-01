from __future__ import annotations

from typing import Any

import numpy as np

from enn._rust import ENNParams as RustENNParams
from enn._rust import ENNStatefulFitter as _RustENNStatefulFitter


class ENNStatefulFitter:
    def __init__(
        self,
        k: int,
        rng: Any,
        *,
        infer_aleatoric_variance_scale: bool = True,
    ) -> None:
        seed = int(rng.integers(0, 2**63 - 1))
        self._rust = _RustENNStatefulFitter(
            k,
            seed,
            infer_aleatoric_variance_scale,
        )

    def tell(
        self,
        x: np.ndarray,
        y: np.ndarray,
        yvar: np.ndarray | None = None,
        y_bounds: np.ndarray | None = None,
    ) -> None:
        x_array = np.asarray(x, dtype=float)
        y_array = np.asarray(y, dtype=float)
        if y_array.ndim == 1:
            y_array = y_array.reshape(-1, 1)
        yvar_array = None
        if yvar is not None:
            yvar_array = np.asarray(yvar, dtype=float)
            if yvar_array.ndim == 1:
                yvar_array = yvar_array.reshape(-1, 1)
        y_bounds_array = None
        if y_bounds is not None:
            y_bounds_array = np.asarray(y_bounds, dtype=float)
        self._rust.tell(x_array, y_array, yvar_array, y_bounds_array)

    def y_std(self) -> np.ndarray:
        return np.asarray(self._rust.y_std(), dtype=float)

    def ask(
        self,
        model: Any,
        *,
        num_fit_candidates: int | None = None,
        num_fit_samples: int | None = None,
        params_warm_start: Any | None = None,
        affine_calibrate: bool = False,
    ) -> Any:
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
            None if num_fit_candidates is None else int(num_fit_candidates),
            num_fit_samples,
            rust_warm_start,
            affine_calibrate,
        )

        return PyENNParams(
            k_num_neighbors=rust_result.k_num_neighbors,
            epistemic_variance_scale=rust_result.epistemic_variance_scale,
            aleatoric_variance_scale=rust_result.aleatoric_variance_scale,
        )

    @property
    def affine_calibrator(self):
        coeffs = self._rust.affine_coeffs()
        if coeffs is None:
            return None
        from .affine_calibrator import AffineCalibrator

        a, b, c = coeffs
        return AffineCalibrator(
            a=np.asarray(a, dtype=float),
            b=np.asarray(b, dtype=float),
            c=np.asarray(c, dtype=float),
        )

    @affine_calibrator.setter
    def affine_calibrator(self, cal) -> None:
        if cal is None:
            return
        self._rust.set_calibrator_abc(
            np.asarray(cal.a, dtype=float),
            np.asarray(cal.b, dtype=float),
            np.asarray(cal.c, dtype=float),
        )

    def calibrate(self, normal: Any) -> Any:
        if self.affine_calibrator is None:
            return normal
        return self.affine_calibrator.apply(normal)

    def _fit_scales(self, params: Any) -> np.ndarray:
        return np.array(
            [
                params.k_num_neighbors,
                params.epistemic_variance_scale,
                params.aleatoric_variance_scale,
            ],
            dtype=float,
        )

    def posterior(
        self,
        model: Any,
        x: np.ndarray,
        params: Any,
        flags: Any | None = None,
    ) -> Any:
        from .enn_normal import ENNNormal
        from .enn_params import PosteriorFlags

        flags = flags if flags is not None else PosteriorFlags()
        mu, se, se_epi, se_ale, idx = self._rust.posterior_calibrated(
            model.rust_backend,
            np.asarray(x, dtype=float),
            self._fit_scales(params),
            flags.exclude_nearest,
            flags.observation_noise,
        )
        idx_arr = np.asarray(idx, dtype=int) if idx is not None else None
        yb = np.asarray(model.rust_backend.y_bounds, dtype=float)
        return ENNNormal(mu, se, se_epi, se_ale, idx=idx_arr, y_bounds=yb)

    def posterior_function_draw(
        self,
        model: Any,
        x: np.ndarray,
        params: Any,
        *,
        function_seeds: Any,
        flags: Any | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        from .enn_class_support import _to_rust_seeds
        from .enn_params import PosteriorFlags

        flags = flags if flags is not None else PosteriorFlags()
        draws, idx = self._rust.function_draw_calibrated(
            model.rust_backend,
            np.asarray(x, dtype=float),
            self._fit_scales(params),
            _to_rust_seeds(function_seeds),
            flags.exclude_nearest,
            flags.observation_noise,
        )
        n_query = np.asarray(x).shape[0]
        idx_arr = np.array(idx, dtype=int) if idx else np.zeros((n_query, 0), dtype=int)
        return np.asarray(draws, dtype=float), idx_arr

    def sample(
        self,
        model: Any,
        x: np.ndarray,
        params: Any,
        num_samples: int,
        rng: Any,
        flags: Any | None = None,
        clip: float | None = None,
    ) -> np.ndarray:
        post = self.posterior(model, x, params, flags=flags)
        seed = int(rng.integers(0, 2**63 - 1))
        return post.sample(num_samples, seed, clip=clip)
