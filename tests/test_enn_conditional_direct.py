from __future__ import annotations

import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_conditional import (
    compute_conditional_posterior,
    compute_conditional_posterior_draw_internals,
)
from enn.enn.enn_params import ENNParams, PosteriorFlags


def _setup_conditional_posterior_test():
    train_x, train_y = (
        np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float),
        np.array([[0.0], [1.0]], dtype=float),
    )
    enn = EpistemicNearestNeighbors(train_x, train_y)
    x_whatif, y_whatif = (
        np.array([[0.5, 0.5]], dtype=float),
        np.array([[0.5]], dtype=float),
    )
    x_test = np.array([[0.25, 0.25]], dtype=float)
    params = ENNParams(
        k_num_neighbors=2, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.0
    )
    return (
        enn,
        x_whatif,
        y_whatif,
        x_test,
        params,
        PosteriorFlags(),
        np.array([[1.0]], dtype=float),
    )


def test_compute_conditional_posterior_direct():
    enn, x_w, y_w, x_t, p, f, s = _setup_conditional_posterior_test()
    post = compute_conditional_posterior(
        enn, x_w, y_w, x_t, params=p, flags=f, y_scale=s
    )
    assert post.mu.shape == (1, 1) and post.se.shape == (1, 1)


def test_compute_conditional_posterior_draw_internals_direct():
    enn, x_w, y_w, x_t, p, f, s = _setup_conditional_posterior_test()
    internals = compute_conditional_posterior_draw_internals(
        enn, x_w, y_w, x_t, params=p, flags=f, y_scale=s
    )
    assert internals.idx.shape == (1, 2) and internals.mu.shape == (1, 1)


def test_compute_conditional_posterior_empty_neighbors():
    train_x = np.zeros((0, 2), dtype=float)
    train_y = np.zeros((0, 1), dtype=float)
    enn = EpistemicNearestNeighbors(train_x, train_y)
    x_whatif = np.array([[0.5, 0.5]], dtype=float)
    y_whatif = np.array([[0.5]], dtype=float)
    x_test = np.array([[0.25, 0.25]], dtype=float)
    params = ENNParams(
        k_num_neighbors=1, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.0
    )
    flags = PosteriorFlags()
    y_scale = np.array([[1.0]], dtype=float)
    post = compute_conditional_posterior(
        enn, x_whatif, y_whatif, x_test, params=params, flags=flags, y_scale=y_scale
    )
    assert post.mu.shape == (1, 1)
    internals = compute_conditional_posterior_draw_internals(
        enn, x_whatif, y_whatif, x_test, params=params, flags=flags, y_scale=y_scale
    )
    assert internals.mu.shape == (1, 1)
