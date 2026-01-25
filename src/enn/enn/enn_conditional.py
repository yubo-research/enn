from __future__ import annotations
import numpy as np
from .candidates import Candidates
from .conditional_posterior_draw_internals import ConditionalPosteriorDrawInternals
from .enn_like_protocol import ENNLike
from .enn_params import ENNParams, PosteriorFlags
from .neighbors import Neighbors

_ENNLike = ENNLike
_Candidates = Candidates
_Neighbors = Neighbors


def _pairwise_sq_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    aa = np.sum(a * a, axis=1, keepdims=True)
    bb = np.sum(b * b, axis=1, keepdims=True).T
    dist2 = aa + bb - 2.0 * (a @ b.T)
    return np.maximum(dist2, 0.0)


def _validate_x(enn: ENNLike, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != enn._num_dim:
        raise ValueError(x.shape)
    return x


def _validate_whatif(
    enn: ENNLike, x_whatif: np.ndarray, y_whatif: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    x_whatif = np.asarray(x_whatif, dtype=float)
    y_whatif = np.asarray(y_whatif, dtype=float)
    if x_whatif.ndim != 2 or x_whatif.shape[1] != enn._num_dim:
        raise ValueError(x_whatif.shape)
    if y_whatif.ndim != 2 or y_whatif.shape[1] != enn._num_metrics:
        raise ValueError(y_whatif.shape)
    if x_whatif.shape[0] != y_whatif.shape[0]:
        raise ValueError((x_whatif.shape, y_whatif.shape))
    return x_whatif, y_whatif


def _scale_x_if_needed(enn: ENNLike, x: np.ndarray) -> np.ndarray:
    return x / enn._x_scale if enn._scale_x else x


def _compute_total_n(enn: ENNLike, num_whatif: int, flags: PosteriorFlags) -> int:
    total_n = len(enn) + int(num_whatif)
    if flags.exclude_nearest and total_n <= 1:
        raise ValueError(total_n)
    return total_n


def _compute_search_k(params: ENNParams, flags: PosteriorFlags, total_n: int) -> int:
    return int(
        min(params.k_num_neighbors + (1 if flags.exclude_nearest else 0), total_n)
    )


def _get_train_candidates(enn: ENNLike, x: np.ndarray, *, search_k: int) -> Candidates:
    batch_size = x.shape[0]
    if len(enn) == 0 or search_k == 0:
        return Candidates(
            dist2=np.zeros((batch_size, 0), dtype=float),
            ids=np.zeros((batch_size, 0), dtype=int),
            y=np.zeros((batch_size, 0, enn._num_metrics), dtype=float),
            yvar=(
                np.zeros((batch_size, 0, enn._num_metrics), dtype=float)
                if enn._train_yvar is not None
                else None
            ),
        )
    train_search_k = int(min(search_k, len(enn)))
    dist2_train, idx_train = enn._enn_index.search(
        x, search_k=train_search_k, exclude_nearest=False
    )
    y_train = enn._train_y[idx_train]
    yvar_train = enn._train_yvar[idx_train] if enn._train_yvar is not None else None
    return Candidates(dist2=dist2_train, ids=idx_train, y=y_train, yvar=yvar_train)


def _get_whatif_candidates(
    enn: ENNLike,
    x: np.ndarray,
    x_whatif: np.ndarray,
    y_whatif: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_scaled = _scale_x_if_needed(enn, x)
    x_whatif_scaled = _scale_x_if_needed(enn, x_whatif)
    dist2_whatif = _pairwise_sq_l2(x_scaled, x_whatif_scaled)
    batch_size = x.shape[0]
    y_whatif_batched = np.broadcast_to(
        y_whatif[np.newaxis, :, :], (batch_size, y_whatif.shape[0], y_whatif.shape[1])
    )
    return dist2_whatif, y_whatif_batched


_WhatifCandidateTuple = tuple[np.ndarray, np.ndarray, np.ndarray]


def _merge_candidates(
    enn: ENNLike,
    *,
    train: Candidates,
    whatif: _WhatifCandidateTuple,
) -> Candidates:
    dist2_whatif, ids_whatif, y_whatif_batched = whatif
    dist2_all = np.concatenate([train.dist2, dist2_whatif], axis=1)
    ids_all = np.concatenate([train.ids, ids_whatif], axis=1)
    y_all = np.concatenate([train.y, y_whatif_batched], axis=1)
    if train.yvar is None:
        return Candidates(dist2=dist2_all, ids=ids_all, y=y_all, yvar=None)
    batch_size = dist2_all.shape[0]
    num_whatif = dist2_whatif.shape[1]
    yvar_whatif = np.zeros((batch_size, num_whatif, enn._num_metrics))
    yvar_all = np.concatenate([train.yvar, yvar_whatif], axis=1)
    return Candidates(dist2=dist2_all, ids=ids_all, y=y_all, yvar=yvar_all)


def _select_sorted_candidates(dist2_all: np.ndarray, *, search_k: int) -> np.ndarray:
    batch_size, num_candidates = dist2_all.shape
    if search_k < num_candidates:
        sel = np.argpartition(dist2_all, kth=search_k - 1, axis=1)[:, :search_k]
    else:
        sel = np.broadcast_to(np.arange(num_candidates), (batch_size, num_candidates))
    sel_dist2 = np.take_along_axis(dist2_all, sel, axis=1)
    sel_order = np.argsort(sel_dist2, axis=1)
    return np.take_along_axis(sel, sel_order, axis=1)


def _take_along_axis_3d(a: np.ndarray, idx_2d: np.ndarray) -> np.ndarray:
    return np.take_along_axis(a, idx_2d[:, :, np.newaxis], axis=1)


def _make_empty_normal(enn: ENNLike, batch_size: int):
    from .enn_normal import ENNNormal

    internals = enn._empty_posterior_internals(batch_size)
    return ENNNormal(internals.mu, internals.se)


def _build_candidates(
    enn: ENNLike,
    x: np.ndarray,
    x_whatif: np.ndarray,
    y_whatif: np.ndarray,
    *,
    search_k: int,
) -> Candidates:
    train_candidates = _get_train_candidates(enn, x, search_k=search_k)
    dist2_whatif, y_whatif_batched = _get_whatif_candidates(enn, x, x_whatif, y_whatif)
    n_train = int(len(enn))
    ids_whatif = np.broadcast_to(
        n_train + np.arange(x_whatif.shape[0], dtype=int), dist2_whatif.shape
    )
    return _merge_candidates(
        enn,
        train=train_candidates,
        whatif=(dist2_whatif, ids_whatif, y_whatif_batched),
    )


def _select_effective_neighbors(
    candidates: Candidates,
    *,
    search_k: int,
    k: int,
    exclude_nearest: bool,
) -> Neighbors | None:
    sel = _select_sorted_candidates(candidates.dist2, search_k=search_k)
    if exclude_nearest:
        sel = sel[:, 1:]
    sel = sel[:, : int(min(k, sel.shape[1]))]
    if sel.shape[1] == 0:
        return None
    dist2s = np.take_along_axis(candidates.dist2, sel, axis=1)
    ids = np.take_along_axis(candidates.ids, sel, axis=1)
    y_neighbors = _take_along_axis_3d(candidates.y, sel)
    yvar_neighbors = (
        _take_along_axis_3d(candidates.yvar, sel)
        if candidates.yvar is not None
        else None
    )
    return Neighbors(dist2=dist2s, ids=ids, y=y_neighbors, yvar=yvar_neighbors)


def _compute_mu_se(
    enn: ENNLike,
    neighbors: Neighbors,
    *,
    params: ENNParams,
    flags: PosteriorFlags,
    y_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    stats = enn._compute_weighted_stats(
        neighbors.dist2,
        neighbors.y,
        yvar_neighbors=neighbors.yvar,
        params=params,
        observation_noise=flags.observation_noise,
        y_scale=y_scale,
    )
    return stats.mu, stats.se


def _compute_draw_internals(
    enn: ENNLike,
    neighbors: Neighbors,
    *,
    params: ENNParams,
    flags: PosteriorFlags,
    y_scale: np.ndarray,
) -> ConditionalPosteriorDrawInternals:
    stats = enn._compute_weighted_stats(
        neighbors.dist2,
        neighbors.y,
        yvar_neighbors=neighbors.yvar,
        params=params,
        observation_noise=flags.observation_noise,
        y_scale=y_scale,
    )
    return ConditionalPosteriorDrawInternals(
        idx=neighbors.ids.astype(int, copy=False),
        w_normalized=stats.w_normalized,
        l2=stats.l2,
        mu=stats.mu,
        se=stats.se,
    )


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
    search_k = _compute_search_k(
        params, flags, _compute_total_n(enn, x_whatif.shape[0], flags)
    )
    if search_k == 0:
        return batch_size, search_k, None
    candidates = _build_candidates(enn, x, x_whatif, y_whatif, search_k=search_k)
    neighbors = _select_effective_neighbors(
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

    x = _validate_x(enn, x)
    x_whatif, y_whatif = _validate_whatif(enn, x_whatif, y_whatif)
    if x_whatif.shape[0] == 0:
        return enn.posterior(x, params=params, flags=flags)
    batch_size, search_k, neighbors = _conditional_neighbors_nonempty_whatif(
        enn, x_whatif, y_whatif, x, params=params, flags=flags
    )
    if search_k == 0 or neighbors is None:
        return _make_empty_normal(enn, batch_size)
    mu, se = _compute_mu_se(enn, neighbors, params=params, flags=flags, y_scale=y_scale)
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
) -> ConditionalPosteriorDrawInternals:
    x = _validate_x(enn, x)
    x_whatif, y_whatif = _validate_whatif(enn, x_whatif, y_whatif)
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
    return _compute_draw_internals(
        enn, neighbors, params=params, flags=flags, y_scale=y_scale
    )
