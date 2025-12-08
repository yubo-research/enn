import math

import pytest

from uhd.mvue_accumulator import MVUE


def test_mvue_single_observation():
    acc = MVUE(decay=1.0)
    acc.update(y=10.0, y_var=4.0)
    assert acc.n == 1
    assert acc.mean == 10.0
    assert acc.var == 4.0
    assert acc.se == 2.0


def test_mvue_two_equal_variance_observations():
    acc = MVUE(decay=1.0)
    acc.update(y=8.0, y_var=4.0)
    acc.update(y=12.0, y_var=4.0)
    assert acc.n == 2
    assert acc.mean == 10.0
    assert acc.var == 2.0
    assert acc.se == math.sqrt(2.0)


def test_mvue_precision_weighting():
    acc = MVUE(decay=1.0)
    acc.update(y=10.0, y_var=1.0)
    acc.update(y=20.0, y_var=9.0)
    expected_mean = (10.0 / 1.0 + 20.0 / 9.0) / (1.0 / 1.0 + 1.0 / 9.0)
    expected_var = 1.0 / (1.0 / 1.0 + 1.0 / 9.0)
    assert acc.n == 2
    assert math.isclose(acc.mean, expected_mean)
    assert math.isclose(acc.var, expected_var)
    assert math.isclose(acc.se, math.sqrt(expected_var))


def test_mvue_no_observations_raises():
    acc = MVUE(decay=1.0)
    with pytest.raises(ValueError, match="No observations yet"):
        _ = acc.mean
    with pytest.raises(ValueError, match="No observations yet"):
        _ = acc.var


def test_mvue_zero_variance_raises():
    acc = MVUE(decay=1.0)
    with pytest.raises(ValueError, match="y_var must be positive"):
        acc.update(y=10.0, y_var=0.0)


def test_mvue_negative_variance_raises():
    acc = MVUE(decay=1.0)
    with pytest.raises(ValueError, match="y_var must be positive"):
        acc.update(y=10.0, y_var=-1.0)


def test_mvue_decay_one_matches_original_behavior():
    acc = MVUE(decay=1.0)
    acc.update(y=8.0, y_var=4.0)
    acc.update(y=12.0, y_var=4.0)
    assert acc.mean == 10.0
    assert acc.var == 2.0


def test_mvue_decay_downweights_older_observations():
    decay = 0.5
    omdecay = 1.0 - decay
    acc = MVUE(decay=decay)

    acc.update(y=10.0, y_var=1.0)
    acc.update(y=20.0, y_var=1.0)

    expected_precision = decay * omdecay * 1.0 + omdecay * 1.0
    expected_weighted_y = decay * omdecay * 10.0 + omdecay * 20.0
    expected_mean = expected_weighted_y / expected_precision
    expected_var = 1.0 / expected_precision

    assert math.isclose(acc.mean, expected_mean)
    assert math.isclose(acc.var, expected_var)


def test_mvue_decay_zero_raises():
    with pytest.raises(ValueError, match="decay must be in"):
        MVUE(decay=0.0)


def test_mvue_decay_negative_raises():
    with pytest.raises(ValueError, match="decay must be in"):
        MVUE(decay=-0.5)


def test_mvue_decay_greater_than_one_raises():
    with pytest.raises(ValueError, match="decay must be in"):
        MVUE(decay=1.5)
