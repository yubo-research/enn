from __future__ import annotations

from evals.harness.work_dirs import fresh_work_dir


def test_fit_after_every_add_uses_num_fit_candidates_one():
    import numpy as np

    from enn.enn.enn_class import EpistemicNearestNeighbors
    from enn.enn.enn_fitter import ENNStatefulFitter
    from enn.enn.enn_params import ENNParams

    rng = np.random.default_rng(7)
    x_all = rng.standard_normal((12, 2))
    y_all = rng.standard_normal((12, 1))
    yvar_all = 0.01 * np.ones_like(y_all)

    model = EpistemicNearestNeighbors(
        np.empty((0, 2)),
        np.empty((0, 1)),
        np.empty((0, 1)),
        work_dir=fresh_work_dir(),
    )
    fitter = ENNStatefulFitter(k=3, rng=np.random.default_rng(7))
    params: ENNParams | None = None
    for row_x, row_y, row_yvar in zip(x_all, y_all, yvar_all):
        row_x = row_x.reshape(1, -1)
        row_y = row_y.reshape(1, -1)
        row_yvar = row_yvar.reshape(1, -1)
        model.add(row_x, row_y, row_yvar)
        fitter.tell(row_x, row_y, row_yvar)
        params = fitter.ask(
            model,
            num_fit_candidates=1,
            num_fit_samples=8,
            params_warm_start=params,
        )
    assert isinstance(params, ENNParams)
    assert params.k_num_neighbors == 3
    assert np.isfinite(params.epistemic_variance_scale)


def test_model_add_returns_nothing():
    import numpy as np

    from enn.enn.enn_class import EpistemicNearestNeighbors

    model = EpistemicNearestNeighbors(
        np.empty((0, 2)), np.empty((0, 1)), work_dir=fresh_work_dir()
    )
    assert model.add(np.zeros((1, 2)), np.zeros((1, 1))) is None
    assert len(model) == 1


def test_tell_with_y_bounds_tracks_warped_y_std():
    import numpy as np

    from enn.enn.enn_fitter import ENNStatefulFitter

    rng = np.random.default_rng(0)
    x = rng.standard_normal((12, 2))
    y = 0.05 + 0.9 * rng.random((12, 1))
    bounds = np.array([[0.0, 1.0]], dtype=float)
    z = np.log(y / (1.0 - y))
    want = float(z.std(ddof=0))

    fitter = ENNStatefulFitter(k=3, rng=np.random.default_rng(0))
    fitter.tell(x, y, y_bounds=bounds)
    got = float(fitter.y_std()[0])
    assert abs(got - want) < 1e-12, f"warped y_std: got {got} want {want}"
    assert got > 1.0, f"expected warped std >> natural, got {got}"


def test_tell_rejects_y_outside_y_bounds():
    import numpy as np
    import pytest

    from enn.enn.enn_fitter import ENNStatefulFitter

    fitter = ENNStatefulFitter(k=2, rng=np.random.default_rng(0))
    x = np.array([[0.0, 0.0]], dtype=float)
    y = np.array([[1.5]], dtype=float)
    bounds = np.array([[0.0, 1.0]], dtype=float)
    with pytest.raises(ValueError, match="bound"):
        fitter.tell(x, y, y_bounds=bounds)
