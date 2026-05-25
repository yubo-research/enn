from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.random import Generator

    from enn.enn.enn_class import EpistemicNearestNeighbors
    from enn.enn.enn_params import ENNParams

    from .config.enn_fit_config import ENNFitConfig
    from .config.enn_index_driver import ENNIndexDriver


def mk_enn(
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    k: int,
    yvar_obs: np.ndarray | None = None,
    *,
    fit: ENNFitConfig | None = None,
    scale_x: bool = False,
    index_driver: ENNIndexDriver | None = None,
    rng: Generator | None = None,
    params_warm_start: ENNParams | None = None,
) -> tuple[EpistemicNearestNeighbors | None, ENNParams | None]:
    from enn.enn.enn_class import EpistemicNearestNeighbors
    from enn.enn.enn_params import ENNParams

    from .config.enn_index_driver import ENNIndexDriver

    if index_driver is None:
        index_driver = ENNIndexDriver.FLAT

    x_obs_array = np.asarray(x_obs, dtype=float)
    if x_obs_array.size == 0:
        return None, None
    y_obs_array = np.asarray(y_obs, dtype=float)
    if y_obs_array.size == 0:
        return None, None
    y = y_obs_array.reshape(-1, 1) if y_obs_array.ndim == 1 else y_obs_array
    yvar = None
    if yvar_obs is not None:
        yvar_array = np.asarray(yvar_obs, dtype=float)
        if yvar_array.size > 0:
            yvar = yvar_array.reshape(-1, 1) if yvar_array.ndim == 1 else yvar_array
    enn_model = EpistemicNearestNeighbors(
        x_obs_array,
        y,
        yvar,
        scale_x=scale_x,
        index_driver=index_driver,
    )
    if len(enn_model) == 0:
        return None, None
    fitted_params: ENNParams | None = None
    if fit is not None and fit.num_fit_samples is not None and rng is not None:
        from enn.enn.enn_fitter import ENNStatefulFitter

        fitter = ENNStatefulFitter(
            k=k,
            rng=rng,
            infer_aleatoric_variance_scale=fit.infer_aleatoric_variance_scale,
        )
        fitter.tell(x_obs_array, y, yvar)
        fitted_params = fitter.ask(
            enn_model,
            num_fit_candidates=(
                fit.num_fit_candidates if fit.num_fit_candidates is not None else 30
            ),
            num_fit_samples=fit.num_fit_samples,
            params_warm_start=params_warm_start,
        )
    else:
        fitted_params = ENNParams(
            k_num_neighbors=k,
            epistemic_variance_scale=1.0,
            aleatoric_variance_scale=0.0,
        )
    return enn_model, fitted_params
