from __future__ import annotations


def test_enn_fit_after_every_add_uses_num_fit_candidates_one():
    import numpy as np

    from enn.enn.enn_class import EpistemicNearestNeighbors
    from enn.enn.enn_fit import ENNIncrementalDelta, enn_fit
    from enn.enn.enn_fitter import ENNStatefulFitter
    from enn.enn.enn_params import ENNParams

    rng = np.random.default_rng(7)
    x_all = rng.standard_normal((12, 2))
    y_all = rng.standard_normal((12, 1))
    yvar_all = 0.01 * np.ones_like(y_all)

    model = EpistemicNearestNeighbors(
        np.empty((0, 2)), np.empty((0, 1)), np.empty((0, 1))
    )
    fitter = ENNStatefulFitter(k=3, rng=np.random.default_rng(7))
    params: ENNParams | None = None
    for row_x, row_y, row_yvar in zip(x_all, y_all, yvar_all):
        row_x = row_x.reshape(1, -1)
        row_y = row_y.reshape(1, -1)
        row_yvar = row_yvar.reshape(1, -1)
        model.add(row_x, row_y, row_yvar)
        params = enn_fit(
            model,
            k=3,
            num_fit_candidates=1,
            num_fit_samples=8,
            rng=np.random.default_rng(7),
            params_warm_start=params,
            incremental=ENNIncrementalDelta(fitter, row_x, row_y, row_yvar),
        )
    assert isinstance(params, ENNParams)
    assert params.k_num_neighbors == 3
    assert np.isfinite(params.epistemic_variance_scale)


def test_enn_fit_incremental_metamorphic_matches_manual_tell_ask():
    import numpy as np

    from enn.enn.enn_class import EpistemicNearestNeighbors
    from enn.enn.enn_fit import ENNIncrementalDelta, enn_fit
    from enn.enn.enn_fitter import ENNStatefulFitter

    rng = np.random.default_rng(99)
    x_all = rng.standard_normal((8, 2))
    y_all = rng.standard_normal((8, 1))

    via_enn_fit = EpistemicNearestNeighbors(np.empty((0, 2)), np.empty((0, 1)))
    fitter_a = ENNStatefulFitter(k=2, rng=np.random.default_rng(99))
    params_a = None

    via_manual = EpistemicNearestNeighbors(np.empty((0, 2)), np.empty((0, 1)))
    fitter_b = ENNStatefulFitter(k=2, rng=np.random.default_rng(99))
    params_b = None

    for row_x, row_y in zip(x_all, y_all):
        row_x = row_x.reshape(1, -1)
        row_y = row_y.reshape(1, -1)
        via_enn_fit.add(row_x, row_y)
        params_a = enn_fit(
            via_enn_fit,
            k=2,
            num_fit_candidates=1,
            num_fit_samples=6,
            rng=np.random.default_rng(99),
            params_warm_start=params_a,
            incremental=ENNIncrementalDelta(fitter_a, row_x, row_y),
        )

        via_manual.add(row_x, row_y)
        fitter_b.tell(row_x, row_y)
        params_b = fitter_b.ask(
            via_manual,
            num_fit_candidates=1,
            num_fit_samples=6,
            params_warm_start=params_b,
        )

    assert params_a is not None and params_b is not None
    assert params_a.k_num_neighbors == params_b.k_num_neighbors
    assert (
        abs(params_a.epistemic_variance_scale - params_b.epistemic_variance_scale)
        < 1e-12
    )
    assert (
        abs(params_a.aleatoric_variance_scale - params_b.aleatoric_variance_scale)
        < 1e-12
    )
