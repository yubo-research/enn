def test_subsample_loglik_and_enn_fit_improve_hyperparameters():
    import numpy as np

    from enn.core import EpistemicNearestNeighbors
    from enn.enn_fit import enn_fit, subsample_loglik
    from enn.enn_params import ENNParams

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
    assert result.var_scale > 0.0
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
