from __future__ import annotations


def test_calculate_sobol_indices_basic():
    import numpy as np

    from enn.enn.enn_util import calculate_sobol_indices

    rng = np.random.default_rng(42)
    n = 50
    d = 3
    x = rng.standard_normal((n, d))
    y = x[:, 0] + 0.1 * rng.standard_normal(n)

    S = calculate_sobol_indices(x, y)
    assert S.shape == (d,)
    assert np.all(S >= 0)
    assert np.all(S <= 1)
    assert S[0] > S[1]
    assert S[0] > S[2]


def test_calculate_sobol_indices_small_n():
    import numpy as np

    from enn.enn.enn_util import calculate_sobol_indices

    rng = np.random.default_rng(42)
    n = 5
    d = 2
    x = rng.standard_normal((n, d))
    y = rng.standard_normal(n)

    S = calculate_sobol_indices(x, y)
    assert S.shape == (d,)
    assert np.all(S == 1.0)


def test_calculate_sobol_indices_y_2d():
    import numpy as np

    from enn.enn.enn_util import calculate_sobol_indices

    rng = np.random.default_rng(42)
    n = 50
    d = 3
    x = rng.standard_normal((n, d))
    y = (x[:, 0] + 0.1 * rng.standard_normal(n)).reshape(-1, 1)

    S = calculate_sobol_indices(x, y)
    assert S.shape == (d,)
    assert np.all(S >= 0)
    assert np.all(S <= 1)


def test_calculate_sobol_indices_zero_variance():
    import numpy as np

    from enn.enn.enn_util import calculate_sobol_indices

    rng = np.random.default_rng(42)
    n = 50
    d = 3
    x = rng.standard_normal((n, d))
    y = np.ones(n)

    S = calculate_sobol_indices(x, y)
    assert S.shape == (d,)
    assert np.all(S == 1.0)


def test_calculate_sobol_indices_low_variance_dimension():
    import numpy as np

    from enn.enn.enn_util import calculate_sobol_indices

    rng = np.random.default_rng(42)
    n = 50
    d = 3
    x = np.zeros((n, d))
    x[:, 0] = rng.standard_normal(n)
    x[:, 1] = 1e-15 * rng.standard_normal(n)
    x[:, 2] = rng.standard_normal(n)
    y = x[:, 0] + x[:, 2] + 0.1 * rng.standard_normal(n)

    S = calculate_sobol_indices(x, y)
    assert S.shape == (d,)
    assert S[1] == 0.0
    assert S[0] > 0
    assert S[2] > 0


def test_calculate_sobol_indices_dtype_preservation():
    import numpy as np

    from enn.enn.enn_util import calculate_sobol_indices

    rng = np.random.default_rng(42)
    n = 50
    d = 3
    x = rng.standard_normal((n, d)).astype(np.float32)
    y = rng.standard_normal(n).astype(np.float32)

    S = calculate_sobol_indices(x, y)
    assert S.shape == (d,)
    assert S.dtype == np.float32
