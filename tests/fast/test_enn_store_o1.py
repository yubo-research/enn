import numpy as np
import pytest

from enn import _rust
from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams, PosteriorFlags
from enn.enn.enn_store import ENNStore, store_name
from evals.harness.work_dirs import fresh_work_dir


def _data(n, d, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.random((n, d))
    y = np.sin(6 * x[:, :1]) + 0.01 * rng.standard_normal((n, 1))
    return x, y


def test_store_name_accepts_enum_and_string():
    assert store_name(ENNStore.O1) == "O1"
    assert store_name("OLOGN") == "OLOGN"
    with pytest.raises(ValueError):
        store_name("O2")


@pytest.mark.parametrize("store", [ENNStore.OLOGN, ENNStore.O1, ENNStore.O1F])
def test_store_posterior_and_draws(store):
    x, y = _data(1500, 3)
    model = EpistemicNearestNeighbors(
        x[:500], y[:500], work_dir=fresh_work_dir(), store=store
    )
    model.add(x[500:], y[500:])
    assert model.store == store
    assert model.weighting == (store != ENNStore.OLOGN)
    params = ENNParams(
        k_num_neighbors=5, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.1
    )
    post = model.posterior(
        x[:20], params=params, flags=PosteriorFlags(observation_noise=True)
    )
    assert np.corrcoef(post.mu.ravel(), y[:20].ravel())[0, 1] > 0.9
    assert np.all(post.se > 0)


@pytest.mark.parametrize("store", [ENNStore.O1, ENNStore.O1F])
def test_o1_store_reopens(store):
    x, y = _data(300, 2, seed=1)
    work_dir = fresh_work_dir()
    EpistemicNearestNeighbors(x, y, work_dir=work_dir, store=store.value)
    reopened = EpistemicNearestNeighbors.open(work_dir)
    assert reopened.store == store
    assert len(reopened) == 300


def test_o1f_metric_change_is_repaired_incrementally():
    x, y = _data(4000, 4, seed=2)
    model = EpistemicNearestNeighbors(x, y, work_dir=fresh_work_dir(), store="O1F")
    backend = model.rust_backend
    _rust.index_set_trees(backend, 8)
    _rust.index_set_repair(backend, 10, 0.25, True)
    backend.set_metric_scale(np.array([0.1, 0.1, 1.0, 1.0]), True)
    backend.set_metric_scale(np.array([1.0, 1.0, 0.1, 0.1]), True)
    assert _rust.index_occupancy(backend)["stale_trees"] == 8
    assert _rust.index_maintain(backend, 5000) == 5000
    occ = _rust.index_occupancy(backend)
    assert (occ["repairs_finished"], occ["repair_cursor"], occ["stale_trees"]) == (
        1,
        1000,
        7,
    )
    assert occ["reachable_rows"] == 4000
    _rust.index_maintain(backend, 8 * 4000)
    assert _rust.index_occupancy(backend)["stale_trees"] == 0
    with pytest.raises(ValueError):
        _rust.index_set_repair(
            EpistemicNearestNeighbors(
                x, y, work_dir=fresh_work_dir(), store="O1"
            ).rust_backend,
            1,
            0.25,
            True,
        )


def test_o1f_pivot_split_rule_keeps_every_row_reachable():
    x, y = _data(2000, 4, seed=3)
    model = EpistemicNearestNeighbors(
        x[:500], y[:500], work_dir=fresh_work_dir(), store="O1F"
    )
    backend = model.rust_backend
    _rust.index_set_split_rule(backend, "pivot")
    model.add(x[500:], y[500:])
    occ = _rust.index_occupancy(backend)
    assert occ["reachable_rows"] == 2000 and occ["splits"] > 0
    _d, idx = backend.neighbor_distances_and_indices(x[:5], 1)
    assert list(idx[:, 0]) == [0, 1, 2, 3, 4]
    _rust.index_set_io_width(backend, 1)
    serial = [
        backend.neighbor_distances_and_indices(x[i : i + 1], 3)[1] for i in range(5)
    ]
    _rust.index_set_io_width(backend, 4, slow_us=0.0)
    for i in range(5):
        assert (
            backend.neighbor_distances_and_indices(x[i : i + 1], 3)[1] == serial[i]
        ).all()
    with pytest.raises(ValueError):
        _rust.index_set_split_rule(backend, "bogus")
    with pytest.raises(ValueError):
        _rust.index_set_split_rule(
            EpistemicNearestNeighbors(
                x, y, work_dir=fresh_work_dir(), store="O1"
            ).rust_backend,
            "pivot",
        )
