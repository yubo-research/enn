from __future__ import annotations

from types import SimpleNamespace

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
    model = EpistemicNearestNeighbors(x, y, scale_x=False)
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

def test_no_surrogate_and_pareto_acq():
    from enn.turbo.python_fallback.components.no_surrogate import NoSurrogate
    from enn.turbo.python_fallback.components.pareto_acq_optimizer import (
        ParetoAcqOptimizer,
    )
    from enn.turbo.python_fallback.components.posterior_result import PosteriorResult

    s = NoSurrogate()
    x = np.array([[0.0, 0.0], [1.0, 1.0]])
    y = np.array([[0.5], [1.5]])
    s.fit(x, y)
    assert s.lengthscales is None
    pred = s.predict(x)
    assert pred.mu.shape[0] == 2
    samples = s.sample(x, num_samples=3, rng=np.random.default_rng(0))
    assert samples.shape[0] == 3
    with pytest.raises(RuntimeError):
        NoSurrogate().predict(x)

    class _Surr:
        def predict(self, x_cand):
            mu = np.column_stack([x_cand[:, 0], x_cand[:, 1]])
            return PosteriorResult(mu=mu, sigma=np.ones_like(mu) * 0.1)

    opt = ParetoAcqOptimizer()
    x_cand = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 0.0], [0.0, 1.0]])
    chosen = opt.select(x_cand, num_arms=2, surrogate=_Surr(), rng=np.random.default_rng(0))
    assert chosen.shape[0] == 2

    class _Surr1d:
        def predict(self, x_cand):
            return PosteriorResult(mu=x_cand[:, 0], sigma=np.ones(x_cand.shape[0]))

    chosen1 = opt.select(
        x_cand, num_arms=1, surrogate=_Surr1d(), rng=np.random.default_rng(1)
    )
    assert chosen1.shape[0] == 1

def test_fallback_strategies_smoke():
    from enn.turbo.python_fallback.strategies.lhd_only_strategy import LHDOnlyStrategy
    from enn.turbo.python_fallback.strategies.turbo_hybrid_strategy import (
        TurboHybridStrategy,
    )

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    rng = np.random.default_rng(0)
    lhd = LHDOnlyStrategy.create(bounds=bounds, rng=rng)
    opt = SimpleNamespace(_rng=rng, _y_tr_list=None)
    x = lhd.ask(opt, num_arms=2)
    assert x.shape == (2, 2)
    y = lhd.tell(opt, SimpleNamespace(y=np.zeros((2, 1))), x_unit=x)
    assert y.shape[0] == 2
    assert lhd.init_progress() is None

    hybrid = TurboHybridStrategy.create(bounds=bounds, rng=np.random.default_rng(1), num_init=4)
    assert hybrid.init_progress() == (0, 4)
    opt_h = SimpleNamespace(
        _rng=np.random.default_rng(2),
        _tr_state=SimpleNamespace(
            needs_restart=lambda: False,
            validate_request=lambda n: None,
            update=lambda *a, **k: None,
            restart=lambda r: None,
        ),
        _x_obs=[],
        _y_obs=[],
        _yvar_obs=[],
        _y_tr_list=[],
        _restart_generation=0,
        _incumbent_idx=None,
        _incumbent_x_unit=None,
        _incumbent_y_scalar=None,
        _incumbent_tracker=SimpleNamespace(reset=lambda: None),
        _ask_normal=lambda n, is_fallback=False: np.zeros((n, 2)),
        _surrogate=SimpleNamespace(
            fit=lambda *a, **k: None,
            predict=lambda x_unit: SimpleNamespace(mu=np.zeros((x_unit.shape[0], 1))),
        ),
        _gp_num_steps=0,
        _dt_fit=0.0,
        _update_incumbent=lambda: None,
    )
    # make _x_obs support len and view like AppendableArray minimally for tell path later
    class _Arr:
        def __init__(self):
            self._a = np.zeros((0, 2))

        def __len__(self):
            return self._a.shape[0]

        def view(self):
            return self._a

    opt_h._x_obs = _Arr()
    opt_h._y_obs = _Arr()
    opt_h._yvar_obs = _Arr()
    xh = hybrid.ask(opt_h, num_arms=2)
    assert xh.shape[0] == 2

def test_optimizer_fixture_capture_smoke():
    from optimizer_fixtures.capture import build_fixture
    from optimizer_fixtures.catalog import FIXTURE_GENERATOR_ENTRIES

    entry = FIXTURE_GENERATOR_ENTRIES[0]
    payload = build_fixture(entry, seed=0)
    assert "steps" in payload
    assert payload["seed"] == 0

def test_thompson_and_ucb_and_selectors():
    from enn.turbo.python_fallback.components.chebyshev_incumbent_selector import (
        ChebyshevIncumbentSelector,
    )
    from enn.turbo.python_fallback.components.no_incumbent_selector import (
        NoIncumbentSelector,
    )
    from enn.turbo.python_fallback.components.posterior_result import PosteriorResult
    from enn.turbo.python_fallback.components.scalar_incumbent_selector import (
        ScalarIncumbentSelector,
    )
    from enn.turbo.python_fallback.components.thompson_acq_optimizer import (
        ThompsonAcqOptimizer,
    )
    from enn.turbo.python_fallback.components.ucb_acq_optimizer import UCBAcqOptimizer

    class _S:
        def predict(self, x):
            mu = np.asarray(x[:, :1], dtype=float)
            return PosteriorResult(mu=mu, sigma=np.ones_like(mu) * 0.1)

        def sample(self, x, num_samples, rng):
            base = x[:, :1]
            return np.broadcast_to(base, (num_samples, x.shape[0], 1)).copy()

    x = np.linspace(0, 1, 8).reshape(-1, 1)
    rng = np.random.default_rng(0)
    assert ThompsonAcqOptimizer().select(x, 2, _S(), rng).shape[0] == 2
    assert UCBAcqOptimizer().select(x, 2, _S(), rng).shape[0] == 2

    y = np.array([[0.1], [0.5], [0.2]])
    mu = y.copy()
    rng = np.random.default_rng(0)
    NoIncumbentSelector().reset(rng)
    assert isinstance(NoIncumbentSelector().select(y, mu, rng), int)
    s = ScalarIncumbentSelector(noise_aware=False)
    s.reset(np.random.default_rng(1))
    idx = s.select(y, mu, np.random.default_rng(1))
    assert isinstance(idx, int)
    c = ChebyshevIncumbentSelector(num_metrics=1, noise_aware=False, alpha=0.05)
    c.reset(np.random.default_rng(2))
    _ = c.select(y, mu, np.random.default_rng(2))

def test_incumbent_tracker_and_gp_surrogate_smoke():
    from enn.turbo.python_fallback.components.incumbent_tracker import (
        build_incumbent_tracker,
    )

    from enn.turbo.config import GPSurrogateConfig, TurboTRConfig
    from enn.turbo.python_fallback.turbo_trust_region import TurboTrustRegion
    from enn.turbo.python_fallback.components.scalar_incumbent_selector import (
        ScalarIncumbentSelector as _S,
    )
    tr_state = TurboTrustRegion(
        config=TurboTRConfig(), num_dim=2, incumbent_selector=_S(noise_aware=False)
    )
    tr = build_incumbent_tracker(GPSurrogateConfig(), tr_state)
    tr.reset()
    x = np.array([[0.0, 0.0], [1.0, 1.0]])
    y = np.array([[0.0], [1.0]])
    if hasattr(tr, 'update'):
        tr.update(x, y)

def test_coverage_enn_fit_fast():
    from enn.enn.enn_class import EpistemicNearestNeighbors
    from enn.enn.enn_fit import ENNIncrementalDelta, enn_fit, subsample_loglik
    from enn.enn.enn_fitter import ENNStatefulFitter
    from enn.enn.enn_params import ENNParams

    rng = np.random.default_rng(8)
    x = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5], [0.2, 0.8]])
    y = np.array([[0.0], [1.0], [0.5], [0.3]])
    model = EpistemicNearestNeighbors(x, y, scale_x=False)
    params = enn_fit(model, k=2, num_fit_candidates=2, num_fit_samples=2, rng=rng)
    assert params.k_num_neighbors >= 1
    fitter = ENNStatefulFitter(k=2, rng=np.random.default_rng(9))
    fitter.tell(x[:2], y[:2])
    model2 = EpistemicNearestNeighbors(x[:2], y[:2], scale_x=False)
    model2.add(x[2:3], y[2:3])
    params2 = enn_fit(
        model2,
        k=2,
        num_fit_candidates=2,
        num_fit_samples=2,
        rng=np.random.default_rng(10),
        incremental=ENNIncrementalDelta(fitter=fitter, x=x[2:3], y=y[2:3]),
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

