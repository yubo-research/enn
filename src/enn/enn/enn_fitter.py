from __future__ import annotations

from typing import Any

import numpy as np

from enn._rust import ENNStatefulFitter as _RustENNStatefulFitter
from enn._rust import subsample_loglik as _rust_subsample_loglik

from .enn_class_support import rust_params


def _py_calibrator(coefficients: Any) -> Any:
    if coefficients is None:
        return None
    from .affine_calibrator import AffineCalibrator

    a, b, c = coefficients
    return AffineCalibrator(
        a=np.asarray(a, dtype=float),
        b=np.asarray(b, dtype=float),
        c=np.asarray(c, dtype=float),
    )


def py_params(rust_result: Any) -> Any:
    from .enn_params import ENNParams as PyENNParams

    return PyENNParams(
        k_num_neighbors=rust_result.k_num_neighbors,
        epistemic_variance_scale=rust_result.epistemic_variance_scale,
        aleatoric_variance_scale=rust_result.aleatoric_variance_scale,
        calibrator=_py_calibrator(rust_result.calibrator),
        tau=rust_result.tau,
    )


def subsample_loglik(
    model: Any,
    x: np.ndarray,
    y: np.ndarray,
    *,
    paramss: list[Any],
    P: int | None = None,
    seed: int,
    y_std: np.ndarray | None = None,
) -> list[float]:
    from .enn_class import EpistemicNearestNeighbors as PyENN

    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)

    if not isinstance(model, PyENN):
        raise TypeError(f"Expected EpistemicNearestNeighbors, got {type(model)}")

    k_values = [p.k_num_neighbors for p in paramss]
    epi_scales = [p.epistemic_variance_scale for p in paramss]
    ale_scales = [p.aleatoric_variance_scale for p in paramss]
    taus = [float(getattr(p, "tau", 0.0)) for p in paramss]

    y_std_arr = None
    if y_std is not None:
        y_std_arr = np.asarray(y_std, dtype=float).ravel()

    return _rust_subsample_loglik(
        model.rust_backend,
        x_array,
        y_array,
        k_values,
        epi_scales,
        ale_scales,
        P,
        int(seed),
        y_std_arr,
        taus,
    )


class ENNStatefulFitter:
    def __init__(
        self,
        k: int,
        *,
        seed: int,
        infer_aleatoric_variance_scale: bool = True,
        infer_prior_precision: bool = False,
    ) -> None:
        self._k = int(k)
        self._seed = int(seed)
        self._rust = _RustENNStatefulFitter(
            self._k,
            self._seed,
            infer_aleatoric_variance_scale,
            infer_prior_precision,
        )

    @property
    def k(self) -> int:
        return self._k

    @property
    def seed(self) -> int:
        return self._seed

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

        if not isinstance(model, PyENN):
            raise TypeError(f"Expected EpistemicNearestNeighbors, got {type(model)}")

        rust_result = self._rust.ask(
            model.rust_backend,
            None if num_fit_candidates is None else int(num_fit_candidates),
            num_fit_samples,
            rust_params(params_warm_start),
            affine_calibrate,
        )
        return py_params(rust_result)
