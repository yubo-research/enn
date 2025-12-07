import math

import pytest

from uhd.mvue_accumulator import MVUE


def test_mvue_single_observation():
    acc = MVUE()
    acc.update(y=10.0, y_var=4.0)
    assert acc.n == 1
    assert acc.mean == 10.0
    assert acc.var == 4.0
    assert acc.se == 2.0


def test_mvue_two_equal_variance_observations():
    acc = MVUE()
    acc.update(y=8.0, y_var=4.0)
    acc.update(y=12.0, y_var=4.0)
    assert acc.n == 2
    assert acc.mean == 10.0
    assert acc.var == 2.0
    assert acc.se == math.sqrt(2.0)


def test_mvue_precision_weighting():
    acc = MVUE()
    acc.update(y=10.0, y_var=1.0)
    acc.update(y=20.0, y_var=9.0)
    expected_mean = (10.0 / 1.0 + 20.0 / 9.0) / (1.0 / 1.0 + 1.0 / 9.0)
    expected_var = 1.0 / (1.0 / 1.0 + 1.0 / 9.0)
    assert acc.n == 2
    assert math.isclose(acc.mean, expected_mean)
    assert math.isclose(acc.var, expected_var)
    assert math.isclose(acc.se, math.sqrt(expected_var))


def test_mvue_no_observations_raises():
    acc = MVUE()
    with pytest.raises(ValueError, match="No observations yet"):
        _ = acc.mean
    with pytest.raises(ValueError, match="No observations yet"):
        _ = acc.var


def test_mvue_zero_variance_raises():
    acc = MVUE()
    with pytest.raises(ValueError, match="y_var must be positive"):
        acc.update(y=10.0, y_var=0.0)


def test_mvue_negative_variance_raises():
    acc = MVUE()
    with pytest.raises(ValueError, match="y_var must be positive"):
        acc.update(y=10.0, y_var=-1.0)
