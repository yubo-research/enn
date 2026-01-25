import numpy as np
import pytest
from enn.turbo.hypervolume import hypervolume_2d_max


@pytest.mark.parametrize(
    "y,expected",
    [
        (np.array([[1.0, 0.5], [0.5, 1.0]]), 0.75),
        (
            np.array([[1.0, 1.0], [0.2, 0.2], [0.5, 0.5]]),
            1.0,
        ),
    ],
)
def test_hypervolume_2d_max(y, expected):
    ref = np.array([0.0, 0.0])
    assert hypervolume_2d_max(y, ref) == expected
