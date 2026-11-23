import numpy as np
import pytest

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


@pytest.mark.parametrize("store", [ENNStore.OLOGN, ENNStore.O1])
def test_store_posterior_and_draws(store):
    x, y = _data(1500, 3)
    model = EpistemicNearestNeighbors(
        x[:500], y[:500], work_dir=fresh_work_dir(), store=store
    )
    model.add(x[500:], y[500:])
    assert model.store == store
    assert model.weighting == (store == ENNStore.O1)
    params = ENNParams(
        k_num_neighbors=5, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.1
    )
    post = model.posterior(
        x[:20], params=params, flags=PosteriorFlags(observation_noise=True)
    )
    assert np.corrcoef(post.mu.ravel(), y[:20].ravel())[0, 1] > 0.9
    assert np.all(post.se > 0)


def test_o1_store_reopens():
    x, y = _data(300, 2, seed=1)
    work_dir = fresh_work_dir()
    EpistemicNearestNeighbors(x, y, work_dir=work_dir, store="O1")
    reopened = EpistemicNearestNeighbors.open(work_dir)
    assert reopened.store == ENNStore.O1
    assert len(reopened) == 300
