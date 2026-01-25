from __future__ import annotations
import numpy as np
import pytest
from enn.enn.enn_index import ENNIndex
from enn.enn.enn_hash import (
    normal_hash_batch_multi_seed,
    normal_hash_batch_multi_seed_fast,
)


def test_enn_index_init_and_search():
    rng = np.random.default_rng(42)
    train_x = rng.standard_normal((20, 3))
    x_scale = np.ones((1, 3), dtype=float)
    index = ENNIndex(train_x, num_dim=3, x_scale=x_scale, scale_x=False)
    query = rng.standard_normal((5, 3))
    dist2s, idx = index.search(query, search_k=3, exclude_nearest=False)
    assert dist2s.shape == (5, 3) and idx.shape == (5, 3)
    assert np.all(idx >= 0) and np.all(idx < 20)


def test_enn_index_search_exclude_nearest():
    rng = np.random.default_rng(42)
    train_x = rng.standard_normal((20, 3))
    x_scale = np.ones((1, 3), dtype=float)
    index = ENNIndex(train_x, num_dim=3, x_scale=x_scale, scale_x=False)
    query = train_x[:3]
    dist2s_include, idx_include = index.search(query, search_k=3, exclude_nearest=False)
    dist2s_exclude, idx_exclude = index.search(query, search_k=3, exclude_nearest=True)
    assert dist2s_include.shape == (3, 3) and dist2s_exclude.shape == (3, 2)
    assert np.allclose(dist2s_include[:, 0], 0.0, atol=1e-6)


def test_enn_index_with_scaling():
    rng = np.random.default_rng(42)
    train_x = rng.standard_normal((20, 3))
    x_scale = np.array([[2.0, 0.5, 1.0]])
    index = ENNIndex(train_x / x_scale, num_dim=3, x_scale=x_scale, scale_x=True)
    query = rng.standard_normal((5, 3))
    dist2s, idx = index.search(query, search_k=3, exclude_nearest=False)
    assert dist2s.shape == (5, 3) and idx.shape == (5, 3)


@pytest.mark.parametrize("query_shape,search_k", [((5, 3), 0), ((5, 4), 3)])
def test_enn_index_search_invalid_inputs(query_shape, search_k):
    rng = np.random.default_rng(42)
    train_x = rng.standard_normal((20, 3))
    index = ENNIndex(train_x, num_dim=3, x_scale=np.ones((1, 3)), scale_x=False)
    with pytest.raises(ValueError):
        index.search(
            rng.standard_normal(query_shape), search_k=search_k, exclude_nearest=False
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
