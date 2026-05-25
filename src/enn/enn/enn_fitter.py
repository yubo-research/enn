from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from enn._rust import ENNParams as RustENNParams
from enn._rust import ENNStatefulFitter as _RustENNStatefulFitter

if TYPE_CHECKING:
    from numpy.random import Generator

    from .enn_class import EpistemicNearestNeighbors
    from .enn_params import ENNParams


class ENNStatefulFitter:
    def __init__(
        self,
        k: int,
        rng: Generator,
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
        model: EpistemicNearestNeighbors,
        *,
        num_fit_candidates: int,
        num_fit_samples: int,
        params_warm_start: ENNParams | None = None,
    ) -> ENNParams:
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

        return PyENNParams(
            k_num_neighbors=rust_result.k_num_neighbors,
            epistemic_variance_scale=rust_result.epistemic_variance_scale,
            aleatoric_variance_scale=rust_result.aleatoric_variance_scale,
        )
