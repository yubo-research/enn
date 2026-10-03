from __future__ import annotations


import numpy as np
import pytest

def test_enn_reexport_and_fitter_surface():
    import enn.enn.enn as enn_mod
    from enn.enn.enn_class import EpistemicNearestNeighbors
    from enn.enn.enn_fitter import ENNStatefulFitter

    assert enn_mod.DrawInternals is not None
    rng = np.random.default_rng(0)
    fitter = ENNStatefulFitter(k=2, rng=rng)
    x = np.array([[0.0, 0.0], [1.0, 1.0]])
    y = np.array([[0.0], [1.0]])
    fitter.tell(x, y)
    assert fitter.y_std().size >= 1
    model = EpistemicNearestNeighbors(x, y)
    params = fitter.ask(model, num_fit_candidates=2, num_fit_samples=2)
    assert params.k_num_neighbors >= 1

def test_init_strategy_base_subclass():
    from enn.turbo.config.init_strategy_base import InitStrategy

    class Dummy(InitStrategy):
        def create_runtime_strategy(self, *, bounds, rng, num_init):
            return (bounds, rng, num_init)

    out = Dummy().create_runtime_strategy(
        bounds=np.zeros((2, 2)), rng=np.random.default_rng(1), num_init=2
    )
    assert out[2] == 2
    with pytest.raises(NotImplementedError):
        InitStrategy().create_runtime_strategy(
            bounds=np.zeros((2, 2)), rng=np.random.default_rng(1), num_init=2
        )



def test_optimizer_fixture_capture_smoke():
    from optimizer_fixtures.capture import build_fixture
    from optimizer_fixtures.catalog import FIXTURE_GENERATOR_ENTRIES

    entry = FIXTURE_GENERATOR_ENTRIES[0]
    payload = build_fixture(entry, seed=0)
    assert "steps" in payload
    assert payload["seed"] == 0



def test_coverage_enn_fit_fast():
    from enn.enn.enn_class import EpistemicNearestNeighbors
    from enn.enn.enn_fit import enn_fit, subsample_loglik
    from enn.enn.enn_fitter import ENNStatefulFitter
    from enn.enn.enn_params import ENNParams

    rng = np.random.default_rng(8)
    x = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5], [0.2, 0.8]])
    y = np.array([[0.0], [1.0], [0.5], [0.3]])
    model = EpistemicNearestNeighbors(x, y)
    params = enn_fit(model, k=2, num_fit_candidates=2, num_fit_samples=2, rng=rng)
    assert params.k_num_neighbors >= 1
    fitter = ENNStatefulFitter(k=2, rng=np.random.default_rng(9))
    fitter.tell(x[:2], y[:2])
    model2 = EpistemicNearestNeighbors(x[:2], y[:2])
    token = model2.add(x[2:3], y[2:3])
    params2 = enn_fit(
        model2,
        k=2,
        num_fit_candidates=2,
        num_fit_samples=2,
        rng=np.random.default_rng(10),
        incremental=token,
        params_warm_start=params,
    )
    assert params2.k_num_neighbors >= 1
    scores = subsample_loglik(
        model,
        x,
        y.ravel(),
        paramss=[
            ENNParams(
                k_num_neighbors=2,
                epistemic_variance_scale=1.0,
                aleatoric_variance_scale=1.0,
            )
        ],
        P=2,
        rng=np.random.default_rng(11),
        y_std=np.ones(1),
    )
    assert len(scores) == 1

