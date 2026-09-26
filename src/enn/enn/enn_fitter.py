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
        self._rng = rng
        self._rust = _RustENNStatefulFitter(
            k,
            seed,
            infer_aleatoric_variance_scale,
        )
        self.affine_calibrator = None

    def tell(
        self,
        x: np.ndarray,
        y: np.ndarray,
        yvar: np.ndarray | None = None,
        y_bounds: np.ndarray | None = None,
    ) -> None:
        self.affine_calibrator = None
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
        num_fit_candidates: int,
        num_fit_samples: int,
        params_warm_start: Any | None = None,
        affine_calibrate: bool = False,
    ) -> Any:
        from .affine_calibrator import fit_affine_calibrator
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

        params = PyENNParams(
            k_num_neighbors=rust_result.k_num_neighbors,
            epistemic_variance_scale=rust_result.epistemic_variance_scale,
            aleatoric_variance_scale=rust_result.aleatoric_variance_scale,
        )
        if affine_calibrate:
            self.affine_calibrator = fit_affine_calibrator(
                model,
                params,
                num_samples=num_fit_samples,
                rng=self._rng,
            )
        else:
            self.affine_calibrator = None
        return params

    def calibrate(self, normal: Any) -> Any:
        if self.affine_calibrator is None:
            return normal
        return self.affine_calibrator.apply(normal)

    def posterior(
        self,
        model: Any,
        x: np.ndarray,
        params: Any,
        flags: Any | None = None,
    ) -> Any:
        if flags is None:
            raw = model.posterior(x, params=params)
        else:
            raw = model.posterior(x, params=params, flags=flags)
        return self.calibrate(raw)

    def posterior_function_draw(
        self,
        model: Any,
        x: np.ndarray,
        params: Any,
        *,
        function_seeds: Any,
        flags: Any | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        draws, idx = model.posterior_function_draw(
            x,
            params,
            function_seeds=function_seeds,
            flags=flags,
        )
        if self.affine_calibrator is None:
            return draws, idx
        if flags is None:
            raw_post = model.posterior(x, params=params)
        else:
            raw_post = model.posterior(x, params=params, flags=flags)
        yb = np.asarray(model.rust_backend.y_bounds, dtype=float)
        mapped = np.transpose(draws, (0, 2, 1))
        se_p = self.affine_calibrator.c.reshape(1, -1) * np.asarray(
            raw_post.se, dtype=float
        )
        mapped = self.affine_calibrator.map_draws(
            mapped,
            raw_post.mu,
            y_bounds=yb,
            se=se_p,
        )
        return np.transpose(mapped, (0, 2, 1)), idx

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
        return post.sample(num_samples, rng=rng, clip=clip)
