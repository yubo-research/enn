from __future__ import annotations

import numpy as np
import pytest

from evals.harness.work_dirs import fresh_work_dir


def test_fitter_surface():
    from enn.enn.enn_class import EpistemicNearestNeighbors
    from enn.enn.enn_fitter import ENNStatefulFitter

    rng = np.random.default_rng(0)
    fitter = ENNStatefulFitter(k=2, seed=int(rng.integers(2**63 - 1)))
    x = np.array([[0.0, 0.0], [1.0, 1.0]])
    y = np.array([[0.0], [1.0]])
    fitter.tell(x, y)
    assert fitter.y_std().size >= 1
    model = EpistemicNearestNeighbors(x, y, work_dir=fresh_work_dir())
    params = fitter.ask(model, num_fit_candidates=2, num_fit_samples=2)
    assert params.k_num_neighbors >= 1


def test_init_strategy_base_subclass():
    from enn.turbo.config.init_strategy_base import InitStrategy

    class Dummy(InitStrategy):
        def create_runtime_strategy(self, *, bounds, num_init):
            return (bounds, num_init)

    out = Dummy().create_runtime_strategy(bounds=np.zeros((2, 2)), num_init=2)
    assert out[1] == 2
    with pytest.raises(NotImplementedError):
        InitStrategy().create_runtime_strategy(bounds=np.zeros((2, 2)), num_init=2)


def test_optimizer_fixture_capture_smoke():
    from optimizer_fixtures.capture import build_fixture
    from optimizer_fixtures.catalog import FIXTURE_GENERATOR_ENTRIES

    entry = FIXTURE_GENERATOR_ENTRIES[0]
    payload = build_fixture(entry, seed=0)
    assert "steps" in payload
    assert payload["seed"] == 0


def test_coverage_enn_fit_fast():
    from enn.enn.enn_class import EpistemicNearestNeighbors
    from enn.enn.enn_fitter import ENNStatefulFitter, subsample_loglik
    from enn.enn.enn_params import ENNParams

    rng = np.random.default_rng(8)
    x = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5], [0.2, 0.8]])
    y = np.array([[0.0], [1.0], [0.5], [0.3]])
    model = EpistemicNearestNeighbors(x[:0], y[:0], work_dir=fresh_work_dir())
    model.add(x, y)
    fitter = ENNStatefulFitter(k=2, seed=int(rng.integers(2**63 - 1)))
    fitter.tell(x, y)
    params = fitter.ask(model, num_fit_candidates=2, num_fit_samples=2)
    assert params.k_num_neighbors >= 1
    fitter2 = ENNStatefulFitter(
        k=2, seed=int(np.random.default_rng(9).integers(2**63 - 1))
    )
    fitter2.tell(x[:2], y[:2])
    model2 = EpistemicNearestNeighbors(x[:2], y[:2], work_dir=fresh_work_dir())
    model2.add(x[2:3], y[2:3])
    fitter2.tell(x[2:3], y[2:3])
    params2 = fitter2.ask(
        model2,
        num_fit_candidates=2,
        num_fit_samples=2,
        params_warm_start=params,
    )
    assert params2.k_num_neighbors >= 1
    scores = subsample_loglik(
        model,
        x,
        y,
        paramss=[
            ENNParams(
                k_num_neighbors=2,
                epistemic_variance_scale=1.0,
                aleatoric_variance_scale=1.0,
            )
        ],
        P=2,
        seed=int(np.random.default_rng(11).integers(2**63 - 1)),
        y_std=np.ones(1),
    )
    assert len(scores) == 1
