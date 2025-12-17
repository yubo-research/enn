import numpy as np

from enn.turbo.hypervolume import hypervolume_2d_max


def test_hypervolume_2d_max_simple_union():
    ref = np.array([0.0, 0.0])
    y = np.array([[1.0, 0.5], [0.5, 1.0]])
    hv = hypervolume_2d_max(y, ref)
    assert hv == 0.75


def test_hypervolume_2d_max_ignores_dominated_points():
    ref = np.array([0.0, 0.0])
    y = np.array([[1.0, 1.0], [0.2, 0.2], [0.5, 0.5]])
    hv = hypervolume_2d_max(y, ref)
    assert hv == 1.0
