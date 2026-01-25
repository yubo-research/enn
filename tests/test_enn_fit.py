def test_subsample_loglik_and_enn_fit_improve_hyperparameters():
    import numpy as np
    from enn.enn.enn_class import EpistemicNearestNeighbors
    from enn.enn.enn_fit import enn_fit, subsample_loglik
    from enn.enn.enn_params import ENNParams

    rng = np.random.default_rng(0)
    x = rng.standard_normal((40, 2))
    y = (x @ np.array([1.5, -0.5]) + 0.1 * rng.standard_normal(40)).reshape(-1, 1)
    model = EpistemicNearestNeighbors(x, y, 0.01 * np.ones_like(y))
    result = enn_fit(
        model,
        k=10,
        num_fit_candidates=30,
        num_fit_samples=20,
        rng=np.random.default_rng(1),
    )
    assert (
        isinstance(result, ENNParams)
        and result.k_num_neighbors == 10
        and result.epistemic_variance_scale > 0.0
    )
    tuned_ll = subsample_loglik(
        model, x, y[:, 0], paramss=[result], P=20, rng=np.random.default_rng(2)
    )[0]
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
    from enn.enn.enn_class import EpistemicNearestNeighbors
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
    assert result.k_num_neighbors == 5
    assert result.epistemic_variance_scale > 0.0
    assert result.aleatoric_variance_scale >= 0.0


def test_enn_fit_with_warm_start():
    import numpy as np
    from enn.enn.enn_class import EpistemicNearestNeighbors
    from enn.enn.enn_fit import enn_fit
    from enn.enn.enn_params import ENNParams

    rng = np.random.default_rng(42)
    n = 30
    d = 2
    x, y, yvar = _make_linear_1d_regression_data(
        rng=rng, n=n, d=d, noise_std=0.1, yvar=0.01
    )
    model = EpistemicNearestNeighbors(x, y, yvar)
    result1 = enn_fit(
        model,
        k=5,
        num_fit_candidates=20,
        num_fit_samples=10,
        rng=rng,
    )
    result2 = enn_fit(
        model,
        k=5,
        num_fit_candidates=20,
        num_fit_samples=10,
        rng=rng,
        params_warm_start=result1,
    )
    assert isinstance(result2, ENNParams)
    assert result2.k_num_neighbors == 5
    assert result2.epistemic_variance_scale > 0.0
    assert result2.aleatoric_variance_scale >= 0.0


def test_enn_fit_supports_multioutput_y():
    import numpy as np
    from enn.enn.enn_class import EpistemicNearestNeighbors
    from enn.enn.enn_fit import enn_fit, subsample_loglik
    from enn.enn.enn_params import ENNParams

    rng = np.random.default_rng(123)
    x = rng.standard_normal((60, 3))
    y1 = x @ [1.0, -2.0, 0.5] + 0.1 * rng.standard_normal(60)
    y2 = np.sin(x @ [-0.5, 0.25, 1.25]) + 0.3 * rng.standard_normal(60)
    y = np.column_stack([y1, y2]).astype(float)
    model = EpistemicNearestNeighbors(x, y, np.ones_like(y) * [[0.01, 0.09]])
    params = enn_fit(
        model,
        k=12,
        num_fit_candidates=40,
        num_fit_samples=25,
        rng=np.random.default_rng(456),
    )
    assert isinstance(params, ENNParams) and params.k_num_neighbors == 12
    lls = subsample_loglik(
        model, x, y, paramss=[params], P=25, rng=np.random.default_rng(789)
    )
    assert len(lls) == 1 and np.isfinite(lls[0])
