from __future__ import annotations

import numpy as np

from .enn_conditional_helpers import (
    build_candidates,
    compute_draw_internals,
    compute_mu_se,
    compute_search_k,
    compute_total_n,
    make_empty_normal,
    select_effective_neighbors,
    validate_whatif,
    validate_x,
)
from .enn_like_protocol import ENNLike
from .enn_params import ENNParams, PosteriorFlags
from .neighbors import Neighbors


def _conditional_neighbors_nonempty_whatif(
    enn: ENNLike,
    x_whatif: np.ndarray,
    y_whatif: np.ndarray,
    x: np.ndarray,
    *,
    params: ENNParams,
    flags: PosteriorFlags,
) -> tuple[int, int, Neighbors | None]:
    batch_size = x.shape[0]
    search_k = compute_search_k(
        params, flags, compute_total_n(enn, x_whatif.shape[0], flags)
    )
    if search_k == 0:
        return batch_size, search_k, None
    candidates = build_candidates(enn, x, x_whatif, y_whatif, search_k=search_k)
    neighbors = select_effective_neighbors(
        candidates,
        search_k=search_k,
        k=params.k_num_neighbors,
        exclude_nearest=flags.exclude_nearest,
    )
    return batch_size, search_k, neighbors


def _compute_conditional_posterior_impl(
    enn: ENNLike,
    x_whatif: np.ndarray,
    y_whatif: np.ndarray,
    x: np.ndarray,
    *,
    params: ENNParams,
    flags: PosteriorFlags,
    y_scale: np.ndarray,
):
    from .enn_normal import ENNNormal

    x = validate_x(enn, x)
    x_whatif, y_whatif = validate_whatif(enn, x_whatif, y_whatif)
    if x_whatif.shape[0] == 0:
        return enn.posterior(x, params=params, flags=flags)
    batch_size, search_k, neighbors = _conditional_neighbors_nonempty_whatif(
        enn, x_whatif, y_whatif, x, params=params, flags=flags
    )
    if search_k == 0 or neighbors is None:
        return make_empty_normal(enn, batch_size)
    mu, se = compute_mu_se(enn, neighbors, params=params, flags=flags, y_scale=y_scale)
    return ENNNormal(mu, se)


def compute_conditional_posterior(
    enn: ENNLike,
    x_whatif: np.ndarray,
    y_whatif: np.ndarray,
    x: np.ndarray,
    *,
    params: ENNParams,
    flags: PosteriorFlags,
    y_scale: np.ndarray,
):
    return _compute_conditional_posterior_impl(
        enn, x_whatif, y_whatif, x, params=params, flags=flags, y_scale=y_scale
    )


def compute_conditional_posterior_draw_internals(
    enn: ENNLike,
    x_whatif: np.ndarray,
    y_whatif: np.ndarray,
    x: np.ndarray,
    *,
    params: ENNParams,
    flags: PosteriorFlags,
    y_scale: np.ndarray,
):
    from .conditional_posterior_draw_internals import ConditionalPosteriorDrawInternals

    x = validate_x(enn, x)
    x_whatif, y_whatif = validate_whatif(enn, x_whatif, y_whatif)
    if x_whatif.shape[0] == 0:
        raise ValueError("x_whatif must be non-empty for conditional draw internals")
    batch_size, search_k, neighbors = _conditional_neighbors_nonempty_whatif(
        enn, x_whatif, y_whatif, x, params=params, flags=flags
    )
    if search_k == 0 or neighbors is None:
        empty_internals = enn._empty_posterior_internals(batch_size)
        return ConditionalPosteriorDrawInternals(
            idx=empty_internals.idx,
            w_normalized=empty_internals.w_normalized,
            l2=empty_internals.l2,
            mu=empty_internals.mu,
            se=empty_internals.se,
        )
    return compute_draw_internals(
        enn, neighbors, params=params, flags=flags, y_scale=y_scale
    )
