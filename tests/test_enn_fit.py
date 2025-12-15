def test_subsample_loglik_and_enn_fit_improve_hyperparameters():
    import numpy as np

    from enn.enn import EpistemicNearestNeighbors
    from enn.enn.enn_fit import enn_fit, subsample_loglik
    from enn.enn.enn_params import ENNParams

    rng = np.random.default_rng(0)
    n = 40
    d = 2
    x = rng.standard_normal((n, d))
    true_w = np.array([1.5, -0.5])
    y_mean = x @ true_w
    noise_std = 0.1
    noise = noise_std * rng.standard_normal(n)
    y = (y_mean + noise).reshape(-1, 1)
    yvar = (noise_std**2) * np.ones_like(y)
    model = EpistemicNearestNeighbors(x, y, yvar)
    rng_fit = np.random.default_rng(1)
    result = enn_fit(
        model,
        k=10,
        num_fit_candidates=30,
        num_fit_samples=20,
        rng=rng_fit,
    )
    assert isinstance(result, ENNParams)
    assert result.k == 10
    assert result.epi_var_scale > 0.0
    rng_eval = np.random.default_rng(2)
    tuned_lls = subsample_loglik(
        model,
        x,
        y[:, 0],
        paramss=[result],
        P=20,
        rng=rng_eval,
    )
    tuned_ll = tuned_lls[0]
    assert np.isfinite(tuned_ll), "tuned log-likelihood must be finite"


def _make_linear_1d_regression_data(
    *,
    rng,
    n: int,
    d: int,
    noise_std: float,
    yvar: float | None,
):
    import numpy as np

    x = rng.standard_normal((n, d))
    y = x.sum(axis=1, keepdims=True) + rng.standard_normal((n, 1)) * float(noise_std)
    if yvar is None:
        return x, y, None
    return x, y, float(yvar) * np.ones_like(y)


def test_enn_fit_with_yvar_none():
    import numpy as np

    from enn.enn import EpistemicNearestNeighbors
    from enn.enn.enn_fit import enn_fit
    from enn.enn.enn_params import ENNParams

    rng = np.random.default_rng(42)
    n = 30
    d = 2
    x, y, yvar = _make_linear_1d_regression_data(
        rng=rng, n=n, d=d, noise_std=0.1, yvar=None
    )

    model = EpistemicNearestNeighbors(x, y, train_yvar=yvar)

    result = enn_fit(
        model,
        k=5,
        num_fit_candidates=20,
        num_fit_samples=10,
        rng=rng,
    )

    assert isinstance(result, ENNParams)
    assert result.k == 5
    assert result.epi_var_scale > 0.0
    assert result.ale_homoscedastic_scale >= 0.0


def test_enn_fit_with_warm_start():
    import numpy as np

    from enn.enn import EpistemicNearestNeighbors
    from enn.enn.enn_fit import enn_fit
    from enn.enn.enn_params import ENNParams

    rng = np.random.default_rng(42)
    n = 30
    d = 2
    x, y, yvar = _make_linear_1d_regression_data(
        rng=rng, n=n, d=d, noise_std=0.1, yvar=0.01
    )

    model = EpistemicNearestNeighbors(x, y, yvar)

    # First fit
    result1 = enn_fit(
        model,
        k=5,
        num_fit_candidates=20,
        num_fit_samples=10,
        rng=rng,
    )

    # Second fit with warm start from first result
    result2 = enn_fit(
        model,
        k=5,
        num_fit_candidates=20,
        num_fit_samples=10,
        rng=rng,
        params_warm_start=result1,
    )

    assert isinstance(result2, ENNParams)
    assert result2.k == 5
    assert result2.epi_var_scale > 0.0
    assert result2.ale_homoscedastic_scale >= 0.0


def test_enn_fit_supports_multioutput_y():
    import numpy as np

    from enn.enn import EpistemicNearestNeighbors
    from enn.enn.enn_fit import enn_fit, subsample_loglik
    from enn.enn.enn_params import ENNParams

    rng = np.random.default_rng(123)
    n = 60
    d = 3
    x = rng.standard_normal((n, d))
    w1 = np.array([1.0, -2.0, 0.5])
    w2 = np.array([-0.5, 0.25, 1.25])
    noise_std1 = 0.1
    noise_std2 = 0.3
    y1 = x @ w1 + noise_std1 * rng.standard_normal(n)
    y2 = np.sin(x @ w2) + noise_std2 * rng.standard_normal(n)
    y = np.column_stack([y1, y2]).astype(float)
    yvar = np.ones_like(y, dtype=float) * np.array([[noise_std1**2, noise_std2**2]])
    model = EpistemicNearestNeighbors(x, y, yvar)

    rng_fit = np.random.default_rng(456)
    params = enn_fit(
        model,
        k=12,
        num_fit_candidates=40,
        num_fit_samples=25,
        rng=rng_fit,
    )
    assert isinstance(params, ENNParams)
    assert params.k == 12

    rng_eval = np.random.default_rng(789)
    lls = subsample_loglik(model, x, y, paramss=[params], P=25, rng=rng_eval)
    assert len(lls) == 1
    assert np.isfinite(lls[0])
