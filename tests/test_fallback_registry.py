from __future__ import annotations

import numpy as np

from enn import create_optimizer, turbo_enn_config, turbo_one_config
from dataclasses import replace

from enn.turbo.config import (
    CandidateGenConfig,
    ENNFitConfig,
    ENNSurrogateConfig,
    MorboTRConfig,
    MultiObjectiveConfig,
    turbo_zero_config,
)
from enn.turbo.fallback_registry import (
    FALLBACK_REGISTRY,
    FallbackEntry,
    fallback_reason,
    requires_python_optimizer_fallback,
)
from enn.turbo.rust_optimizer import RustOptimizer, is_rust_supported_config


def test_registry_entries():
    assert FallbackEntry.__dataclass_fields__
    ids = {e.id for e in FALLBACK_REGISTRY}
    assert ids == {"gpsurrogate_turbo_one"}


def test_gpsurrogate_fallback():
    config = turbo_one_config(num_init=2)
    assert requires_python_optimizer_fallback(config)
    assert fallback_reason(config) == "gpsurrogate_turbo_one"
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    opt = create_optimizer(bounds=bounds, config=config, rng=np.random.default_rng(0))
    assert not isinstance(opt, RustOptimizer)


def test_morbo_routes_to_rust():
    config = turbo_enn_config(
        enn=ENNSurrogateConfig(k=3, fit=ENNFitConfig(num_fit_samples=10)),
        trust_region=MorboTRConfig(multi_objective=MultiObjectiveConfig(num_metrics=2)),
    )
    assert fallback_reason(config) is None
    assert is_rust_supported_config(config)
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    opt = create_optimizer(bounds=bounds, config=config, rng=np.random.default_rng(1))
    assert isinstance(opt, RustOptimizer)


def test_num_candidates_per_arm_routes_to_rust():
    base = turbo_zero_config(num_init=3)
    config = replace(
        base,
        candidates=CandidateGenConfig(num_candidates_per_arm=100),
    )
    assert fallback_reason(config) is None
    assert is_rust_supported_config(config)


def test_default_num_candidates_routes_to_rust():
    config = turbo_zero_config(num_init=3)
    assert config.candidates.num_candidates is None
    assert fallback_reason(config) is None
    assert is_rust_supported_config(config)
