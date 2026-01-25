from __future__ import annotations

import numpy as np
import pytest

from enn.turbo.tr_helpers import (
    ScalarIncumbentMixin,
    get_incumbent_index,
    get_scalar_incumbent_value,
    get_single_incumbent_index,
)


class MockSelector:
    def __init__(self, noise_aware=False):
        self.noise_aware = noise_aware

    def select(self, y, mu, rng):
        return np.argmax(mu if self.noise_aware else y)


def test_get_single_incumbent_index():
    selector = MockSelector()
    y = np.array([1.0, 3.0, 2.0])
    rng = np.random.default_rng(0)
    idx = get_single_incumbent_index(selector, y, rng)
    assert np.array_equal(idx, [1])
    assert get_single_incumbent_index(selector, np.array([]), rng).size == 0


def test_get_incumbent_index():
    selector = MockSelector()
    y = np.array([1.0, 3.0, 2.0])
    rng = np.random.default_rng(0)
    assert get_incumbent_index(selector, y, rng) == 1
    with pytest.raises(ValueError, match="empty"):
        get_incumbent_index(selector, np.array([]), rng)


def test_get_scalar_incumbent_value():
    selector = MockSelector(noise_aware=True)
    y = np.array([1.0, 3.0, 2.0])
    mu = np.array([1.5, 2.5, 2.1])
    rng = np.random.default_rng(0)
    val = get_scalar_incumbent_value(selector, y, rng, mu_obs=mu)
    assert val[0] == 2.5
    assert get_scalar_incumbent_value(selector, np.array([]), rng).size == 0


def test_scalar_incumbent_mixin():
    class TestMixin(ScalarIncumbentMixin):
        def __init__(self, selector):
            self.incumbent_selector = selector

    selector = MockSelector()
    mixin = TestMixin(selector)
    y = np.array([1.0, 3.0, 2.0])
    rng = np.random.default_rng(0)
    assert mixin.get_incumbent_index(y, rng) == 1
    assert mixin.get_incumbent_value(y, rng)[0] == 3.0
