from __future__ import annotations

import numpy as np
import pytest

from enn.turbo.config import (
    ENNSurrogateConfig,
    HybridInit,
    InitConfig,
    LHDOnlyInit,
    RandomAcquisitionConfig,
    turbo_zero_config,
)
from enn.turbo.config.validation import validate_optimizer_config


def test_hybrid_init_runtime_strategy_is_not_a_python_object():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="Rust optimizer"):
        HybridInit().create_runtime_strategy(bounds=bounds, rng=rng, num_init=4)


def test_lhd_only_init_marker_not_python_runtime():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(0)
    with pytest.raises(RuntimeError, match="Rust-routing config marker"):
        LHDOnlyInit().create_runtime_strategy(bounds=bounds, rng=rng, num_init=4)


def test_validate_optimizer_config_lhd_only_requires_no_surrogate_direct_call():
    class Dummy:
        def __init__(self) -> None:
            self.init = InitConfig(init_strategy=LHDOnlyInit())
            self.surrogate = ENNSurrogateConfig()
            self.acquisition = RandomAcquisitionConfig()

    validate_optimizer_config(turbo_zero_config())
    bad = Dummy()
    with pytest.raises(ValueError, match="init_strategy='lhd_only'"):
        validate_optimizer_config(bad)


def test_validate_optimizer_config_rejects_unknown_acquisition():
    class NotAnAcquisition:
        pass

    bad = type("Cfg", (), {})()
    bad.init = InitConfig(init_strategy=LHDOnlyInit())
    bad.surrogate = ENNSurrogateConfig()
    bad.acquisition = NotAnAcquisition()
    with pytest.raises(TypeError, match="AcquisitionConfig"):
        validate_optimizer_config(bad)


def test_optimizer_init_progress_and_telemetry_smoke():
    from enn import create_optimizer

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(0)
    opt = create_optimizer(bounds=bounds, config=turbo_zero_config(num_init=5), rng=rng)
    init = opt.init_progress
    assert init is not None
    init_idx, num_init = init
    assert init_idx == 0 and num_init == 5
    _ = opt.ask(num_arms=2)
    tel = opt.telemetry()
    assert tel.dt_fit >= 0.0
    assert tel.dt_gen >= 0.0
    assert tel.dt_sel >= 0.0


def test_turbo_hybrid_fallback_executes_when_init_points_exhausted_mid_batch():
    from enn import create_optimizer

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
