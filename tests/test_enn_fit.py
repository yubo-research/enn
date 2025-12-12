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


def test_enn_fit_with_yvar_none():
    import numpy as np

    from enn.enn import EpistemicNearestNeighbors
    from enn.enn.enn_fit import enn_fit
    from enn.enn.enn_params import ENNParams

    rng = np.random.default_rng(42)
    n = 30
    d = 2
    x = rng.standard_normal((n, d))
    y = x.sum(axis=1, keepdims=True) + rng.standard_normal((n, 1)) * 0.1

    model = EpistemicNearestNeighbors(x, y, train_yvar=None)

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
    x = rng.standard_normal((n, d))
    y = x.sum(axis=1, keepdims=True) + rng.standard_normal((n, 1)) * 0.1
    yvar = 0.01 * np.ones_like(y)

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
