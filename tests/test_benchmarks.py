import numpy as np
from enn.benchmarks import Ackley, DoubleAckley


def test_ackley_1d_input():
    rng = np.random.default_rng(42)
    ackley = Ackley(noise=0.0, rng=rng)
    x = np.zeros(5)
    result = ackley(x)
    assert result.shape == (1,)


def test_ackley_2d_input():
    rng = np.random.default_rng(42)
    ackley = Ackley(noise=0.0, rng=rng)
    x = np.zeros((10, 5))
    result = ackley(x)
    assert result.shape == (10,)


def test_ackley_optimum_near_one():
    rng = np.random.default_rng(42)
    ackley = Ackley(noise=0.0, rng=rng)
    x_opt = np.ones((1, 5))
    x_off = np.zeros((1, 5))
    y_opt = ackley(x_opt)[0]
    y_off = ackley(x_off)[0]
    assert y_opt > y_off


def test_ackley_deterministic_with_same_rng():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    ackley1 = Ackley(noise=0.1, rng=rng1)
    ackley2 = Ackley(noise=0.1, rng=rng2)
    x = np.random.default_rng(0).random((5, 3))
    np.testing.assert_array_equal(ackley1(x), ackley2(x))


def test_ackley_bounds():
    rng = np.random.default_rng(42)
    ackley = Ackley(noise=0.0, rng=rng)
    assert ackley.bounds == [-32.768, 32.768]


def test_double_ackley_shape():
    rng = np.random.default_rng(42)
    obj = DoubleAckley(noise=0.0, rng=rng)
    x = np.zeros((10, 6))
    result = obj(x)
    assert result.shape == (10, 2)


def test_double_ackley_requires_even_dims():
    rng = np.random.default_rng(42)
    obj = DoubleAckley(noise=0.0, rng=rng)
    x = np.zeros((5, 5))
    try:
        obj(x)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_double_ackley_deterministic():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    obj1 = DoubleAckley(noise=0.1, rng=rng1)
    obj2 = DoubleAckley(noise=0.1, rng=rng2)
    x = np.random.default_rng(0).random((5, 6))
    np.testing.assert_array_equal(obj1(x), obj2(x))
