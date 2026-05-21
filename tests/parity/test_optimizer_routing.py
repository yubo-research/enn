from __future__ import annotations

import numpy as np
import pytest

from enn import create_optimizer, turbo_enn_config, turbo_one_config, turbo_zero_config
from enn.turbo.config import (
    AcqType,
    CandidateGenConfig,
    ENNFitConfig,
    ENNSurrogateConfig,
    MorboTRConfig,
    MultiObjectiveConfig,
    lhd_only_config,
)
from enn.turbo.fallback_registry import FALLBACK_REGISTRY, fallback_reason
from enn.turbo.rust_optimizer import RustOptimizer, is_rust_supported_config
from enn.turbo.rust_optimizer_helpers import DEFAULT_ENN_K, resolve_enn_k

try:
    from enn._rust import Optimizer  # noqa: F401

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

pytestmark = pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust not available")

BOUNDS = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)


def test_default_turbo_enn_routes_to_rust():
    config = turbo_enn_config()
    assert is_rust_supported_config(config)
    opt = create_optimizer(bounds=BOUNDS, config=config, rng=np.random.default_rng(0))
    assert isinstance(opt, RustOptimizer)


def test_turbo_enn_k_none_resolves_default_k():
    config = turbo_enn_config(enn=ENNSurrogateConfig(k=None))
    assert resolve_enn_k(config) == DEFAULT_ENN_K
    opt = create_optimizer(bounds=BOUNDS, config=config, rng=np.random.default_rng(1))
    assert isinstance(opt, RustOptimizer)


@pytest.mark.parametrize("acq_type", [AcqType.UCB, AcqType.THOMPSON])
def test_turbo_enn_acq_types_route_to_rust(acq_type):
    config = turbo_enn_config(
        acq_type=acq_type,
        enn=ENNSurrogateConfig(k=4, fit=ENNFitConfig(num_fit_samples=10)),
    )
    opt = create_optimizer(bounds=BOUNDS, config=config, rng=np.random.default_rng(2))
    assert isinstance(opt, RustOptimizer)


def test_const_num_candidates_routes_to_rust():
    config = turbo_enn_config(
        candidates=CandidateGenConfig(num_candidates=256),
        enn=ENNSurrogateConfig(k=4, fit=ENNFitConfig(num_fit_samples=10)),
        acq_type=AcqType.UCB,
    )
    assert is_rust_supported_config(config)
    opt = create_optimizer(bounds=BOUNDS, config=config, rng=np.random.default_rng(3))
    assert isinstance(opt, RustOptimizer)


def test_turbo_zero_and_lhd_route_to_rust():
    for cfg in (turbo_zero_config(num_init=4), lhd_only_config(num_init=5)):
        opt = create_optimizer(bounds=BOUNDS, config=cfg, rng=np.random.default_rng(4))
        assert isinstance(opt, RustOptimizer)


def test_turbo_one_falls_back_with_registry_reason():
    config = turbo_one_config(num_init=3)
    assert fallback_reason(config) == "gpsurrogate_turbo_one"
    assert not is_rust_supported_config(config)
    opt = create_optimizer(bounds=BOUNDS, config=config, rng=np.random.default_rng(5))
    assert not isinstance(opt, RustOptimizer)


def test_morbo_routes_to_rust():
    config = turbo_enn_config(
        enn=ENNSurrogateConfig(k=3, fit=ENNFitConfig(num_fit_samples=10)),
        trust_region=MorboTRConfig(multi_objective=MultiObjectiveConfig(num_metrics=2)),
    )
    assert fallback_reason(config) is None
    assert is_rust_supported_config(config)
    opt = create_optimizer(bounds=BOUNDS, config=config, rng=np.random.default_rng(6))
    assert isinstance(opt, RustOptimizer)


def test_fallback_registry_has_one_entry():
    assert len(FALLBACK_REGISTRY) == 1


def test_supported_enn_create_does_not_call_python_optimizer(monkeypatch):
    import enn.turbo.python_fallback.optimizer as py_optimizer

    def _fail(**_kwargs):
        raise AssertionError(
            "python Optimizer must not be used for supported ENN config"
        )

    monkeypatch.setattr(py_optimizer, "create_optimizer", _fail)
    config = turbo_enn_config(
        enn=ENNSurrogateConfig(k=4, fit=ENNFitConfig(num_fit_samples=10)),
        num_init=4,
    )
    opt = create_optimizer(bounds=BOUNDS, config=config, rng=np.random.default_rng(7))
    assert isinstance(opt, RustOptimizer)
