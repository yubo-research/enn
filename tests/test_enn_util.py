from __future__ import annotations


def _make_sobol_synth_data(*, rng, n: int, d: int, y_2d: bool) -> tuple:
    x = rng.standard_normal((n, d))
    y = x[:, 0] + 0.1 * rng.standard_normal(n)
    if y_2d:
        y = y.reshape(-1, 1)
    return x, y


def test_calculate_sobol_indices_basic():
    import numpy as np

    from enn.enn.enn_util import calculate_sobol_indices

    rng = np.random.default_rng(42)
    n = 50
    d = 3
    x, y = _make_sobol_synth_data(rng=rng, n=n, d=d, y_2d=False)

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
    x, y = _make_sobol_synth_data(rng=rng, n=n, d=d, y_2d=True)

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


def test_arms_from_pareto_fronts_selects_fronts_in_order():
    import numpy as np

    from enn.enn.enn_util import arms_from_pareto_fronts

    x_cand = np.arange(12, dtype=float).reshape(6, 2)
    mu = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0], dtype=float)
    se = np.array([0.10, 0.20, 0.15, 0.40, 0.05, 0.50], dtype=float)
    rng = np.random.default_rng(0)

    out4 = arms_from_pareto_fronts(x_cand, mu, se, num_arms=4, rng=rng)
    assert out4.shape == (4, 2)
    assert np.allclose(out4, x_cand[[0, 1, 3, 5]])

    rng = np.random.default_rng(0)
    out5 = arms_from_pareto_fronts(x_cand, mu, se, num_arms=5, rng=rng)
    assert out5.shape == (5, 2)
    assert np.allclose(out5, x_cand[[0, 1, 2, 3, 5]])
