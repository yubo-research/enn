from __future__ import annotations
import numpy as np
import pytest
from enn.turbo.config.init_strategies import HybridInit, LHDOnlyInit
from enn.turbo.config.validation import validate_optimizer_config
from enn.turbo.optimizer import Optimizer, create_optimizer
from enn.turbo.optimizer_config import (
    GPSurrogateConfig,
    InitConfig,
    RAASPOptimizerConfig,
    RandomAcquisitionConfig,
    turbo_zero_config,
)
from enn.turbo.sampling import draw_lhd
from enn.turbo.strategies import LHDOnlyStrategy, TurboHybridStrategy


def test_draw_lhd_shapes_and_bounds():
    bounds = np.array([[-1.0, 1.0], [0.0, 2.0]], dtype=float)
    rng = np.random.default_rng(0)
    x = draw_lhd(bounds=bounds, num_arms=32, rng=rng)
    assert x.shape == (32, 2)
    assert np.all(x[:, 0] >= -1.0) and np.all(x[:, 0] <= 1.0)
    assert np.all(x[:, 1] >= 0.0) and np.all(x[:, 1] <= 2.0)


def test_init_strategies_build_runtime_strategies():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(0)
    hybrid = HybridInit().create_runtime_strategy(bounds=bounds, rng=rng, num_init=4)
    assert isinstance(hybrid, TurboHybridStrategy)
    lhd_only = LHDOnlyInit().create_runtime_strategy(bounds=bounds, rng=rng, num_init=4)
    assert isinstance(lhd_only, LHDOnlyStrategy)


def test_validate_optimizer_config_lhd_only_requires_no_surrogate_direct_call():
    class Dummy:
        def __init__(self) -> None:
            self.init = InitConfig(init_strategy=LHDOnlyInit())
            self.surrogate = GPSurrogateConfig()
            self.acquisition = RandomAcquisitionConfig()
            self.acq_optimizer = RAASPOptimizerConfig()

    validate_optimizer_config(turbo_zero_config())
    bad = Dummy()
    with pytest.raises(ValueError, match="init_strategy='lhd_only'"):
        validate_optimizer_config(bad)


def test_optimizer_init_progress_and_telemetry_smoke():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(0)
    opt = create_optimizer(bounds=bounds, config=turbo_zero_config(num_init=5), rng=rng)
    init = opt.init_progress
    assert init is not None
    init_idx, num_init = init
    assert init_idx == 0 and num_init == 5
    _ = opt.ask(num_arms=2)
    tel = opt.telemetry()
    assert tel.dt_fit == 0.0
    assert tel.dt_gen == 0.0
    assert tel.dt_sel == 0.0


def test_turbo_hybrid_fallback_executes_when_init_points_exhausted_mid_batch():
    bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=float)
    rng = np.random.default_rng(0)
    opt = create_optimizer(bounds=bounds, config=turbo_zero_config(num_init=2), rng=rng)
    x1 = opt.ask(num_arms=1)
    y1 = -np.sum(x1**2, axis=1)
    opt.tell(x1, y1)
    init_before = opt.init_progress
    assert init_before is not None
    init_idx_before, num_init = init_before
    assert init_idx_before == 1 and num_init == 2
    x2 = opt.ask(num_arms=2)
    assert x2.shape == (2, 2)


def test_optimizer_direct_constructor_builds_strategy_by_default():
    from enn.turbo.components import NoSurrogate
    from enn.turbo.components import RandomAcqOptimizer

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(0)
    cfg = turbo_zero_config(num_init=3)
    opt = Optimizer(
        bounds=bounds,
        config=cfg,
        rng=rng,
        surrogate=NoSurrogate(),
        acquisition_optimizer=RandomAcqOptimizer(),
    )
    init = opt.init_progress
    assert init is not None
    assert init == (0, 3)
