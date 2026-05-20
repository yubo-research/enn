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


def test_enn_fit_num_fit_samples_sweep_prints_results():
    import json

    import numpy as np

    from enn.enn.enn_class import EpistemicNearestNeighbors
    from enn.enn.enn_fit import enn_fit, subsample_loglik

    rng = np.random.default_rng(20260522)
    x_train = rng.standard_normal((1000, 3))
    y_train = (
        x_train @ np.array([1.5, -0.5, 0.25]) + 0.1 * rng.standard_normal(1000)
    ).reshape(-1, 1)
    y_var_train = 0.01 * np.ones_like(y_train)

    def capture_result(model, params, num_fit_samples, eval_x, eval_y):
        loglik = subsample_loglik(
            model,
            eval_x,
            eval_y,
            paramss=[params],
            P=len(eval_x),
            rng=np.random.default_rng(2000 + num_fit_samples),
        )[0]
        return {
            "num_fit_samples": num_fit_samples,
            "params": {
                "k_num_neighbors": params.k_num_neighbors,
                "epistemic_variance_scale": params.epistemic_variance_scale,
                "aleatoric_variance_scale": params.aleatoric_variance_scale,
            },
            "subsample_loglik": loglik,
        }

    batch_captured = []
    for num_fit_samples in [10, 30, 100, 300, 1000]:
        model = EpistemicNearestNeighbors(x_train, y_train, y_var_train)
        params = enn_fit(
            model,
            k=10,
            num_fit_candidates=100,
            num_fit_samples=num_fit_samples,
            rng=np.random.default_rng(1000 + num_fit_samples),
        )
        batch_captured.append(
            capture_result(model, params, num_fit_samples, x_train, y_train)
        )

    incremental_captured = []
    incremental_model = EpistemicNearestNeighbors(
        np.empty((0, x_train.shape[1])),
        np.empty((0, y_train.shape[1])),
        np.empty((0, y_var_train.shape[1])),
    )
    params_warm_start = None
    for num_fit_samples, (x, y, y_var) in enumerate(
        zip(x_train, y_train, y_var_train), start=1
    ):
        incremental_model.add(
            x.reshape(1, -1),
            y.reshape(1, -1),
            y_var.reshape(1, -1),
        )
        params_warm_start = enn_fit(
            incremental_model,
            k=10,
            num_fit_candidates=1,
            num_fit_samples=100,
            rng=np.random.default_rng(3000 + num_fit_samples),
            params_warm_start=params_warm_start,
        )
        if num_fit_samples in [10, 30, 100, 300, 1000]:
            incremental_captured.append(
                capture_result(
                    incremental_model,
                    params_warm_start,
                    num_fit_samples,
                    x_train[:num_fit_samples],
                    y_train[:num_fit_samples],
                )
            )

    print(
        json.dumps(
            {"batch": batch_captured, "incremental": incremental_captured}, indent=2
        )
    )


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


def test_enn_fit_can_disable_aleatoric_inference():
    import numpy as np

    from enn.enn.enn_class import EpistemicNearestNeighbors
    from enn.enn.enn_fit import enn_fit
    from enn.enn.enn_params import ENNParams

    rng = np.random.default_rng(42)
    x, y, yvar = _make_linear_1d_regression_data(
        rng=rng, n=40, d=3, noise_std=0.2, yvar=0.01
    )
    model = EpistemicNearestNeighbors(x, y, yvar)
    warm = ENNParams(
        k_num_neighbors=5,
        epistemic_variance_scale=1.0,
        aleatoric_variance_scale=123.0,
    )
    result = enn_fit(
        model,
        k=5,
        num_fit_candidates=25,
        num_fit_samples=15,
        rng=rng,
        params_warm_start=warm,
        infer_aleatoric_variance_scale=False,
    )
    assert isinstance(result, ENNParams)
    assert result.k_num_neighbors == 5
    assert result.aleatoric_variance_scale == 0.0
