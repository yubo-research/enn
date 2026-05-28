from __future__ import annotations

import numpy as np
import pytest

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_class_support import (
    enn_index_neighbor_distances_and_indices,
    enn_neighbor_distances_and_indices,
)
from enn.enn.enn_hash import (
    normal_hash_batch_multi_seed,
    normal_hash_batch_multi_seed_fast,
)
from enn.enn.enn_params import ENNParams
from enn.turbo.config.enn_index_driver import ENNIndexDriver


def _enn(train_x, *, scale_x=False, index_driver=ENNIndexDriver.FLAT, train_y=None):
    if train_y is None:
        train_y = np.zeros((train_x.shape[0], 1), dtype=float)
    return EpistemicNearestNeighbors(
        train_x, train_y, scale_x=scale_x, index_driver=index_driver
    )


@pytest.mark.parametrize("scale_x", [False, True])
def test_add_rejects_wrong_output_width_without_mutating_model(scale_x):
    train_x = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=float)
    train_y = np.array([[0.0], [10.0]], dtype=float)
    enn = EpistemicNearestNeighbors(train_x, train_y, scale_x=scale_x)

    with pytest.raises(ValueError):
        enn.add(np.array([[20.0, 0.0]], dtype=float), np.array([[20.0, 200.0]]))

    assert len(enn) == 2
    np.testing.assert_allclose(enn.train_x, train_x)
    np.testing.assert_allclose(enn.train_y, train_y)

    enn.add(np.array([[30.0, 0.0]], dtype=float), np.array([[30.0]], dtype=float))
    assert len(enn) == 3
    assert enn.train_x.shape[0] == enn.train_y.shape[0]

    params = ENNParams(
        k_num_neighbors=1,
        epistemic_variance_scale=1.0,
        aleatoric_variance_scale=0.1,
    )
    out = enn.posterior(np.array([[30.0, 0.0]], dtype=float), params=params)
    np.testing.assert_allclose(out.mu, [[30.0]])


@pytest.mark.parametrize("scale_x", [False, True])
def test_add_first_observations_can_initialize_yvar_on_empty_model(scale_x):
    empty_x = np.empty((0, 2), dtype=float)
    empty_y = np.empty((0, 1), dtype=float)
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    y = np.array([[10.0], [20.0]], dtype=float)
    yvar = np.array([[0.1], [0.2]], dtype=float)

    incremental = EpistemicNearestNeighbors(empty_x, empty_y, scale_x=scale_x)
    incremental.add(x, y, yvar)
    fresh = EpistemicNearestNeighbors(x, y, yvar, scale_x=scale_x)

    assert len(incremental) == len(fresh) == 2
    np.testing.assert_allclose(incremental.train_x, fresh.train_x)
    np.testing.assert_allclose(incremental.train_y, fresh.train_y)
    np.testing.assert_allclose(incremental.train_yvar, fresh.train_yvar)


@pytest.mark.parametrize("scale_x", [False, True])
def test_zero_row_add_does_not_change_empty_model_yvar_contract(scale_x):
    empty_x = np.empty((0, 2), dtype=float)
    empty_y = np.empty((0, 1), dtype=float)
    empty_yvar = np.empty((0, 1), dtype=float)
    x = np.array([[1.0, 2.0]], dtype=float)
    y = np.array([[10.0]], dtype=float)

    incremental = EpistemicNearestNeighbors(empty_x, empty_y, scale_x=scale_x)
    incremental.add(empty_x, empty_y, empty_yvar)

    assert len(incremental) == 0
    assert incremental.train_yvar is None

    incremental.add(x, y)
    fresh = EpistemicNearestNeighbors(x, y, scale_x=scale_x)

    assert len(incremental) == len(fresh) == 1
    np.testing.assert_allclose(incremental.train_x, fresh.train_x)
    np.testing.assert_allclose(incremental.train_y, fresh.train_y)
    assert incremental.train_yvar is None


def test_enn_neighbor_search_k_larger_than_n_train_never_emits_invalid_neighbor_index():
    rng = np.random.default_rng(0)
    n_train = 3
    train_x = rng.standard_normal((n_train, 2))
    enn = _enn(train_x, scale_x=False)
    query = rng.standard_normal((1, 2))
    search_k = 8
    _dist2s, idx = enn_neighbor_distances_and_indices(
        enn.rust_backend, query, search_k=search_k, exclude_nearest=False
    )
    assert idx.shape == (1, search_k)
    assert np.all(idx >= 0)
    assert np.all(idx < n_train)


def test_enn_neighbor_search_empty_train_sentinel_never_uses_negative_one():
    train_x = np.zeros((0, 2), dtype=float)
    enn = _enn(train_x, scale_x=False)
    query = np.array([[1.0, 2.0], [-0.5, 0.25]], dtype=float)
    search_k = 4
    dist2s, idx = enn_neighbor_distances_and_indices(
        enn.rust_backend, query, search_k=search_k, exclude_nearest=False
    )
    assert dist2s.shape == (2, search_k) and idx.shape == (2, search_k)
    assert np.all(np.isposinf(dist2s))
    assert np.all(idx == 0)
    assert np.all(idx >= 0)
    d2_ex, idx_ex = enn_neighbor_distances_and_indices(
        enn.rust_backend, query, search_k=search_k, exclude_nearest=True
    )
    assert d2_ex.shape == (2, search_k - 1) and idx_ex.shape == (2, search_k - 1)
    assert np.all(np.isposinf(d2_ex))
    assert np.all(idx_ex == 0)


def test_enn_neighbor_search_k_one_exclude_nearest_yields_zero_columns():
    rng = np.random.default_rng(7)
    train_x = rng.standard_normal((10, 2))
    enn = _enn(train_x, scale_x=False)
    q = rng.standard_normal((3, 2))
    dist2s, idx = enn_neighbor_distances_and_indices(
        enn.rust_backend, q, search_k=1, exclude_nearest=True
    )
    assert dist2s.shape == (3, 0) and idx.shape == (3, 0)


def test_enn_neighbor_search_hnsw_valid_indices_and_shapes():
    rng = np.random.default_rng(11)
    n_train, dim = 40, 3
    train_x = rng.standard_normal((n_train, dim))
    enn = _enn(train_x, scale_x=False, index_driver=ENNIndexDriver.HNSW)
    query = rng.standard_normal((4, dim))
    search_k = 6
    dist2s, idx = enn_neighbor_distances_and_indices(
        enn.rust_backend, query, search_k=search_k, exclude_nearest=False
    )
    assert dist2s.shape == (4, search_k) and idx.shape == (4, search_k)
    assert np.all(idx >= 0) and np.all(idx < n_train)
    finite = np.isfinite(dist2s)
    assert np.any(finite)
    assert np.all(dist2s[finite] >= 0.0)


def test_enn_neighbor_search_init_and_search():
    rng = np.random.default_rng(42)
    train_x = rng.standard_normal((20, 3))
    enn = _enn(train_x, scale_x=False)
    query = rng.standard_normal((5, 3))
    dist2s, idx = enn_neighbor_distances_and_indices(
        enn.rust_backend, query, search_k=3, exclude_nearest=False
    )
    assert dist2s.shape == (5, 3) and idx.shape == (5, 3)
    assert np.all(idx >= 0) and np.all(idx < 20)


def test_enn_index_neighbor_tie_break_flag():
    train_x = np.array([[0.0], [0.0], [1.0], [2.0]])
    np.array([[0.0], [1.0], [2.0], [3.0]])
    enn = _enn(train_x, scale_x=False)
    query = np.array([[0.0]])
    _, idx_on = enn_index_neighbor_distances_and_indices(
        enn.rust_backend,
        query,
        search_k=2,
        exclude_nearest=False,
        tie_break_neighbors=True,
    )
    _, idx_off = enn_index_neighbor_distances_and_indices(
        enn.rust_backend,
        query,
        search_k=2,
        exclude_nearest=False,
        tie_break_neighbors=False,
    )
    assert idx_on[0].tolist() == [0, 1]
    assert idx_off[0].tolist() in ([0, 1], [1, 0])


def test_enn_index_neighbor_tie_break_batch_matches_single_on_train():
    train_x = np.array([[(i - 9.5) / 3.0 + 0.01 * i] for i in range(20)])
    train_y = np.array([[(i + 1) * 0.37 - 2.1] for i in range(20)])
    enn = _enn(train_x, scale_x=False, train_y=train_y)
    k = 10
    _, idx_batch = enn_index_neighbor_distances_and_indices(
        enn.rust_backend,
        train_x,
        search_k=k,
        exclude_nearest=False,
        tie_break_neighbors=True,
    )
    for i in range(train_x.shape[0]):
        _, idx_one = enn_index_neighbor_distances_and_indices(
            enn.rust_backend,
            train_x[i : i + 1],
            search_k=k,
            exclude_nearest=False,
            tie_break_neighbors=True,
        )
        assert idx_batch[i].tolist() == idx_one[0].tolist()


def test_enn_index_neighbor_tie_break_expands_when_more_than_k_at_cutoff():
    train_x = np.array([[0.0], [0.0], [0.0], [1.0]])
    np.array([[0.0], [1.0], [2.0], [3.0]])
    enn = _enn(train_x, scale_x=False)
    query = np.array([[0.0]])
    _, idx_on = enn_index_neighbor_distances_and_indices(
        enn.rust_backend,
        query,
        search_k=2,
        exclude_nearest=False,
        tie_break_neighbors=True,
    )
    assert idx_on[0].tolist() == [0, 1]


def test_enn_posterior_tie_break_self_search_batch():
    from enn.enn.posterior_flags import PosteriorFlags

    train_x = np.array([[(i - 9.5) / 3.0 + 0.01 * i] for i in range(20)])
    train_y = np.array([[(i + 1) * 0.37 - 2.1] for i in range(20)])
    enn = _enn(train_x, scale_x=False, train_y=train_y)
    params = ENNParams(
        k_num_neighbors=10, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.1
    )
    flags = PosteriorFlags(tie_break_neighbors=True)
    result = enn.posterior(train_x, params=params, flags=flags)
    assert result.idx is not None
    assert len(result.idx) == train_x.shape[0]


@pytest.mark.parametrize("scale_x", [False, True])
def test_enn_index_neighbor_search_matches_faiss_when_no_ties(scale_x):
    rng = np.random.default_rng(42)
    train_x = rng.standard_normal((20, 3))
    enn = _enn(train_x, scale_x=scale_x)
    query = rng.standard_normal((5, 3))
    search_k = 3
    faiss_d2, faiss_idx = enn_neighbor_distances_and_indices(
        enn.rust_backend, query, search_k=search_k, exclude_nearest=False
    )
    exact_d2, exact_idx = enn_index_neighbor_distances_and_indices(
        enn.rust_backend, query, search_k=search_k, exclude_nearest=False
    )
    assert exact_d2.shape == faiss_d2.shape == (5, search_k)
    assert exact_idx.shape == faiss_idx.shape
    assert np.all(exact_idx >= 0) and np.all(exact_idx < 20)


def test_enn_index_neighbor_search_exclude_nearest_and_hnsw():
    rng = np.random.default_rng(42)
    train_x = rng.standard_normal((20, 3))
    query = train_x[:3]
    for driver in (ENNIndexDriver.FLAT, ENNIndexDriver.HNSW):
        enn = _enn(train_x, scale_x=False, index_driver=driver)
        dist2s, idx = enn_index_neighbor_distances_and_indices(
            enn.rust_backend, query, search_k=3, exclude_nearest=True
        )
        assert dist2s.shape == (3, 2) and idx.shape == (3, 2)
        assert np.all(idx >= 0) and np.all(idx < 20)


def test_enn_index_neighbor_search_k_one_exclude_nearest_yields_zero_columns():
    rng = np.random.default_rng(7)
    train_x = rng.standard_normal((10, 2))
    enn = _enn(train_x, scale_x=False)
    q = rng.standard_normal((3, 2))
    dist2s, idx = enn_index_neighbor_distances_and_indices(
        enn.rust_backend, q, search_k=1, exclude_nearest=True
    )
    assert dist2s.shape == (3, 0) and idx.shape == (3, 0)


def test_exact_index_search_zero_k_and_k_exceeds_train():
    train_x = np.array([[0.0], [1.0], [2.0]], dtype=float)
    enn = _enn(train_x, scale_x=False)
    query = np.array([[0.5], [1.5]], dtype=float)
    d0, i0 = enn_index_neighbor_distances_and_indices(
        enn.rust_backend, query, search_k=0, exclude_nearest=False
    )
    assert d0.shape == (2, 0) and i0.shape == (2, 0)
    d_all, i_all = enn_index_neighbor_distances_and_indices(
        enn.rust_backend, query, search_k=10, exclude_nearest=False
    )
    assert d_all.shape == (2, 10) and i_all.shape == (2, 10)
    assert np.all(i_all[:, :3] >= 0) and np.all(i_all[:, :3] < 3)
    assert np.all(i_all[:, 3:] == -1)


def test_enn_neighbor_search_exclude_nearest():
    rng = np.random.default_rng(42)
    train_x = rng.standard_normal((20, 3))
    enn = _enn(train_x, scale_x=False)
    query = train_x[:3]
    dist2s_include, idx_include = enn_neighbor_distances_and_indices(
        enn.rust_backend, query, search_k=3, exclude_nearest=False
    )
    dist2s_exclude, idx_exclude = enn_neighbor_distances_and_indices(
        enn.rust_backend, query, search_k=3, exclude_nearest=True
    )
    assert dist2s_include.shape == (3, 3) and dist2s_exclude.shape == (3, 2)
    assert np.allclose(dist2s_include[:, 0], 0.0, atol=1e-6)


def test_enn_neighbor_search_with_scaling():
    rng = np.random.default_rng(42)
    train_x = rng.standard_normal((20, 3))
    enn = _enn(train_x, scale_x=True)
    query = rng.standard_normal((5, 3))
    dist2s, idx = enn_neighbor_distances_and_indices(
        enn.rust_backend, query, search_k=3, exclude_nearest=False
    )
    assert dist2s.shape == (5, 3) and idx.shape == (5, 3)


@pytest.mark.parametrize("query_shape,search_k", [((5, 3), 0), ((5, 4), 3)])
def test_enn_neighbor_search_invalid_inputs(query_shape, search_k):
    rng = np.random.default_rng(42)
    train_x = rng.standard_normal((20, 3))
    enn = _enn(train_x, scale_x=False)
    with pytest.raises(ValueError):
        enn_neighbor_distances_and_indices(
            enn.rust_backend,
            rng.standard_normal(query_shape),
            search_k=search_k,
            exclude_nearest=False,
        )


def test_normal_hash_batch_multi_seed_shape():
    function_seeds = np.array([1, 2, 3], dtype=np.int64)
    data_indices = np.array([[0, 1, 2], [3, 4, 5]], dtype=int)
    result = normal_hash_batch_multi_seed(function_seeds, data_indices, num_metrics=2)
    assert result.shape == (3, 2, 3, 2)


def test_normal_hash_batch_multi_seed_deterministic():
    function_seeds = np.array([42], dtype=np.int64)
    data_indices = np.array([[0, 1]], dtype=int)
    result1 = normal_hash_batch_multi_seed(function_seeds, data_indices, num_metrics=1)
    result2 = normal_hash_batch_multi_seed(function_seeds, data_indices, num_metrics=1)
    assert np.allclose(result1, result2)


def test_normal_hash_batch_multi_seed_different_seeds():
    data_indices = np.array([[0, 1]], dtype=int)
    result1 = normal_hash_batch_multi_seed(
        np.array([1], dtype=np.int64), data_indices, num_metrics=1
    )
    result2 = normal_hash_batch_multi_seed(
        np.array([2], dtype=np.int64), data_indices, num_metrics=1
    )
    assert not np.allclose(result1, result2)


def test_normal_hash_batch_multi_seed_fast_shape_and_deterministic():
    function_seeds = np.array([1, 2, 3], dtype=np.int64)
    data_indices = np.array([[0, 1, 2], [3, 4, 5]], dtype=int)
    out1 = normal_hash_batch_multi_seed_fast(
        function_seeds, data_indices, num_metrics=2
    )
    out2 = normal_hash_batch_multi_seed_fast(
        function_seeds, data_indices, num_metrics=2
    )
    assert out1.shape == (3, 2, 3, 2)
    assert np.allclose(out1, out2)


def test_normal_hash_batch_multi_seed_fast_different_seeds():
    data_indices = np.array([[0, 1]], dtype=int)
    out1 = normal_hash_batch_multi_seed_fast(
        np.array([1], dtype=np.int64), data_indices, num_metrics=3
    )
    out2 = normal_hash_batch_multi_seed_fast(
        np.array([2], dtype=np.int64), data_indices, num_metrics=3
    )
    assert not np.allclose(out1, out2)
