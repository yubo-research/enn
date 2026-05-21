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
    default_num_candidates,
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
    assert ids == {
        "gpsurrogate_turbo_one",
        "morbo_tr",
        "custom_num_candidates_callable",
    }


def test_gpsurrogate_fallback():
    config = turbo_one_config(num_init=2)
    assert requires_python_optimizer_fallback(config)
    assert fallback_reason(config) == "gpsurrogate_turbo_one"
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    opt = create_optimizer(bounds=bounds, config=config, rng=np.random.default_rng(0))
    assert not isinstance(opt, RustOptimizer)


def test_morbo_fallback():
    config = turbo_enn_config(
        enn=ENNSurrogateConfig(k=3, fit=ENNFitConfig(num_fit_samples=10)),
        trust_region=MorboTRConfig(multi_objective=MultiObjectiveConfig(num_metrics=2)),
    )
    assert fallback_reason(config) == "morbo_tr"
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    opt = create_optimizer(bounds=bounds, config=config, rng=np.random.default_rng(1))
    assert not isinstance(opt, RustOptimizer)


def test_custom_num_candidates_callable_fallback():
    def varying(*, num_dim: int, num_arms: int) -> int:
        return num_dim * 10

    base = turbo_zero_config(num_init=3)
    config = replace(
        base,
        candidates=CandidateGenConfig(num_candidates=varying),
    )
    assert fallback_reason(config) == "custom_num_candidates_callable"
    assert not is_rust_supported_config(config)


def test_default_num_candidates_not_fallback():
    config = turbo_zero_config(num_init=3)
    assert config.candidates.num_candidates is default_num_candidates
    assert fallback_reason(config) is None
    assert is_rust_supported_config(config)
