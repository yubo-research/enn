import numpy as np

from enn.enn.draw_internals import DrawInternals
from enn.enn.neighbor_data import NeighborData
from enn.enn.weighted_stats import WeightedStats


def test_draw_internals():
    di = DrawInternals(
        idx=np.array([[0, 1]]),
        w_normalized=np.array([[[0.5], [0.5]]]),
        l2=np.array([[1.0]]),
        mu=np.array([[0.5]]),
        se=np.array([[0.1]]),
        se_epi=np.array([[0.1]]),
        se_ale=np.array([[0.0]]),
    )
    assert di.idx.shape == (1, 2)
    assert di.mu.shape == (1, 1)


def test_neighbor_data():
    nd = NeighborData(
        dist2s=np.array([[0.1, 0.2]]),
        idx=np.array([[0, 1]]),
        y_neighbors=np.array([[[1.0], [2.0]]]),
        k=2,
    )
    assert nd.k == 2
    assert nd.idx.shape == (1, 2)


def test_weighted_stats():
    ws = WeightedStats(
        w_normalized=np.array([[[0.5], [0.5]]]),
        l2=np.array([[1.0]]),
        mu=np.array([[0.5]]),
        se=np.array([[0.1]]),
        se_epi=np.array([[0.1]]),
        se_ale=np.array([[0.0]]),
    )
    assert ws.mu.shape == (1, 1)
