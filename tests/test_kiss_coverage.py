from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Shared helpers (reduce duplication across config/trust-region tests)
# ---------------------------------------------------------------------------


def _morbo_tr_config():
    from enn.turbo.config import MorboTRConfig, MultiObjectiveConfig

    mo = MultiObjectiveConfig(num_metrics=2, alpha=0.05)
    return MorboTRConfig(multi_objective=mo)


def _morbo_trust_region():
    from enn.turbo.python_fallback.morbo_trust_region import MorboTrustRegion

    cfg = _morbo_tr_config()
    return MorboTrustRegion(config=cfg, num_dim=3, rng=np.random.default_rng(42))


def _turbo_trust_region():
    from enn.turbo.config import TurboTRConfig
    from enn.turbo.python_fallback.turbo_trust_region import TurboTrustRegion

    tr = TurboTrustRegion(config=TurboTRConfig(), num_dim=5)
    tr.validate_request(4)
    return tr


def _enn_model():
    from enn.enn.enn_class import EpistemicNearestNeighbors

    x = np.array([[0.1, 0.2], [0.3, 0.4]])
    y = np.array([[1.0], [2.0]])
    return EpistemicNearestNeighbors(x, y, scale_x=False)


def _optimizer():
    from enn import create_optimizer
    from enn.turbo.config import turbo_zero_config

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    return create_optimizer(
        bounds=bounds, config=turbo_zero_config(), rng=np.random.default_rng(0)
    )


# ---------------------------------------------------------------------------
# Config properties
# ---------------------------------------------------------------------------


def test_morbo_tr_config_rescalarize():
    from enn.turbo.config import (
        MorboTRConfig,
        MultiObjectiveConfig,
        Rescalarize,
        RescalePolicyConfig,
    )

    mo = MultiObjectiveConfig(num_metrics=2, alpha=0.05)
    cfg = MorboTRConfig(multi_objective=mo, rescale_policy=RescalePolicyConfig())
    assert cfg.rescalarize == Rescalarize.ON_PROPOSE


def test_morbo_tr_config_properties():
    cfg = _morbo_tr_config()
    # num_metrics
    assert cfg.num_metrics == 2
    # alpha
    assert cfg.alpha == 0.05
    # length_init / length_min / length_max
    assert isinstance(cfg.length_init, float)
    assert isinstance(cfg.length_min, float)
    assert isinstance(cfg.length_max, float)


def test_turbo_tr_config_properties():
    from enn.turbo.config import TurboTRConfig

    cfg = TurboTRConfig()
    # length_init / length_min / length_max
    assert isinstance(cfg.length_init, float)
    assert isinstance(cfg.length_min, float)
    assert isinstance(cfg.length_max, float)


def test_enn_surrogate_config_properties():
    from enn.turbo.config import ENNFitConfig, ENNSurrogateConfig

    cfg = ENNSurrogateConfig(
        fit=ENNFitConfig(num_fit_samples=50, num_fit_candidates=30)
    )
    # num_fit_samples / num_fit_candidates
    assert cfg.num_fit_samples == 50
    assert cfg.num_fit_candidates == 30


def test_observation_history_config_empty():
    from enn.turbo.config.observation_history_config import ObservationHistoryConfig

    cfg = ObservationHistoryConfig()
    assert cfg == ObservationHistoryConfig()


def test_trust_region_config_protocol():
    from typing import get_args

    from enn.turbo.config.trust_region import InitStrategy, TrustRegionConfig

    assert get_args(TrustRegionConfig)
    assert hasattr(InitStrategy, "create_runtime_strategy")


def test_enn_index_driver_enum():
    from enn.turbo.config.enn_index_driver import ENNIndexDriver

    assert ENNIndexDriver.FLAT != ENNIndexDriver.HNSW


def test_num_candidates_fn_protocol():
    from enn.turbo.config.num_candidates_fn import NumCandidatesFn

    assert NumCandidatesFn is not None


def test_optimizer_config_properties():
    from enn.turbo.config import MorboTRConfig, MultiObjectiveConfig, OptimizerConfig

    cfg = OptimizerConfig()
    # num_metrics
    assert cfg.num_metrics is None
    mo = MultiObjectiveConfig(num_metrics=3, alpha=0.05)
    cfg2 = OptimizerConfig(trust_region=MorboTRConfig(multi_objective=mo))
    assert cfg2.num_metrics == 3
    # candidate_rv / raasp_driver
    assert cfg.candidate_rv is not None
    assert cfg.raasp_driver is not None


# ---------------------------------------------------------------------------
# Trust region properties
# ---------------------------------------------------------------------------


def test_morbo_trust_region_properties():
    tr = _morbo_trust_region()
    # num_dim / num_metrics / length / rescalarize
    from enn.turbo.config import Rescalarize

    assert tr.num_dim == 3
    assert tr.num_metrics == 2
    assert isinstance(tr.length, float)
    assert tr.rescalarize == Rescalarize.ON_PROPOSE


def test_turbo_trust_region_properties():
    tr = _turbo_trust_region()
    # length_init / length_min / length_max / num_metrics / failure_tolerance
    assert isinstance(tr.length_init, float)
    assert isinstance(tr.length_min, float)
    assert isinstance(tr.length_max, float)
    assert tr.num_metrics == 1
    assert isinstance(tr.failure_tolerance, int)


def test_no_trust_region_num_metrics():
    from enn.turbo.config import NoTRConfig
    from enn.turbo.python_fallback.no_trust_region import NoTrustRegion

    tr = NoTrustRegion(config=NoTRConfig(), num_dim=3)
    assert tr.num_metrics == 1


# ---------------------------------------------------------------------------
# Component protocols / properties
# ---------------------------------------------------------------------------


def test_surrogate_protocol_properties():
    from enn.turbo.python_fallback.components.protocols import Surrogate

    # lengthscales / find_x_center
    assert hasattr(Surrogate, "lengthscales")
    assert hasattr(Surrogate, "find_x_center")


def test_trust_region_protocol_properties():
    from enn.turbo.python_fallback.components.protocols import TrustRegion

    # length / compute_bounds
    assert hasattr(TrustRegion, "length")
    assert hasattr(TrustRegion, "compute_bounds")


def test_acquisition_optimizer_protocol():
    from enn.turbo.python_fallback.components.protocols import AcquisitionOptimizer

    assert hasattr(AcquisitionOptimizer, "select")


def test_surrogate_lengthscales():
    from enn.turbo.python_fallback.components.gp_surrogate import GPSurrogate

    assert GPSurrogate().lengthscales is None


def test_incumbent_selector_protocol():
    from enn.turbo.python_fallback.components.incumbent_selector_protocol import (
        IncumbentSelector,
    )

    assert hasattr(IncumbentSelector, "select")


def test_thompson_acq_optimizer_class():
    from enn.turbo.python_fallback.components.thompson_acq_optimizer import (
        ThompsonAcqOptimizer,
    )

    t = ThompsonAcqOptimizer()
    assert hasattr(t, "select")


def test_pareto_and_random_acq_optimizer_select():
    from enn.turbo.python_fallback.components.pareto_acq_optimizer import (
        ParetoAcqOptimizer,
    )
    from enn.turbo.python_fallback.components.random_acq_optimizer import (
        RandomAcqOptimizer,
    )

    rng = np.random.default_rng(0)
    surrogate = _fit_gp_surrogate_for_kiss(rng)
    x_cand = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=float)
    assert ParetoAcqOptimizer().select(x_cand, 2, surrogate, rng).shape == (2, 2)
    assert RandomAcqOptimizer().select(x_cand, 2, surrogate, rng).shape == (2, 2)


def _fit_gp_surrogate_for_kiss(rng):
    from enn.turbo.python_fallback.components.gp_surrogate import GPSurrogate

    surrogate = GPSurrogate()
    x = np.array([[0.2, 0.3], [0.5, 0.5], [0.7, 0.8]], dtype=float)
    y = np.array([0.5, 0.7, 0.3], dtype=float)
    surrogate.fit(x, y, None, num_steps=5, rng=rng)
    return surrogate


# ---------------------------------------------------------------------------
# ENN class/index
# ---------------------------------------------------------------------------


def test_enn_class_properties():
    enn = _enn_model()
    # train_y / train_yvar / num_outputs
    assert enn.train_y.shape == (2, 1)
    assert enn.train_yvar is None
    assert enn.num_outputs == 1


def test_enn_class_add():
    enn = _enn_model()
    enn.add(np.array([[0.5, 0.6]]), np.array([[3.0]]))
    assert len(enn) == 3


def test_enn_neighbor_distances_add_and_search():
    from enn.enn.enn_class_support import enn_neighbor_distances_and_indices

    enn = _enn_model()
    enn.add(np.array([[0.5, 0.6]], dtype=float), np.array([[3.0]], dtype=float))
    _d2, nn = enn_neighbor_distances_and_indices(
        enn.rust_backend,
        np.array([[0.5, 0.6]], dtype=float),
        search_k=3,
        exclude_nearest=False,
    )
    assert nn.shape[1] == 3


# ---------------------------------------------------------------------------
# Optimizer properties
# ---------------------------------------------------------------------------


def test_optimizer_properties():
    opt = _optimizer()
    # tr_obs_count / tr_length
    assert isinstance(opt.tr_obs_count, int)
    assert isinstance(opt.tr_length, float)


def test_optimizer_init_progress():
    opt = _optimizer()
    opt.ask(2)
    assert opt.init_progress is not None


# ---------------------------------------------------------------------------
# Strategy classes
# ---------------------------------------------------------------------------


def test_lhd_only_strategy():
    from enn.turbo.python_fallback.strategies.lhd_only_strategy import LHDOnlyStrategy

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    s = LHDOnlyStrategy.create(bounds=bounds, rng=np.random.default_rng(0))
    assert isinstance(s, LHDOnlyStrategy)
    assert s.init_progress() is None


def test_optimization_strategy_protocol():
    from enn.turbo.python_fallback.strategies.optimization_strategy import (
        OptimizationStrategy,
    )

    assert hasattr(OptimizationStrategy, "ask")
    assert hasattr(OptimizationStrategy, "init_progress")


def test_turbo_hybrid_strategy():
    from enn.turbo.python_fallback.strategies.turbo_hybrid_strategy import (
        TurboHybridStrategy,
    )

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    s = TurboHybridStrategy.create(
        bounds=bounds, rng=np.random.default_rng(0), num_init=4
    )
    assert isinstance(s, TurboHybridStrategy)
    assert s.init_progress() == (0, 4)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def test_build_trust_region():
    from enn.turbo.python_fallback.components.builder import build_trust_region
    from enn.turbo.config import NoTRConfig, TurboTRConfig

    rng = np.random.default_rng(0)
    tr = build_trust_region(TurboTRConfig(), num_dim=3, rng=rng)
    assert hasattr(tr, "length")
    tr2 = build_trust_region(NoTRConfig(), num_dim=3, rng=rng)
    assert hasattr(tr2, "length")


def test_turbo_gp_base():
    from enn.turbo.python_fallback.turbo_gp_base import TurboGPBase

    assert hasattr(TurboGPBase, "forward")


def test_scalar_incumbent_mixin():
    from enn.turbo.python_fallback.turbo_utils import ScalarIncumbentMixin

    assert hasattr(ScalarIncumbentMixin, "get_incumbent_index")


def test_lazy_getattr():
    from enn._lazy import lazy_getattr

    mapping = {"foo": (".enn.enn_params", "ENNParams")}
    result = lazy_getattr(
        name="foo",
        module_name="enn",
        package="enn",
        mapping=mapping,
        extra="pip install ennbo[with-deps]",
    )
    from enn.enn.enn_params import ENNParams

    assert result is ENNParams


def test_lazy_getattr_missing():
    from enn._lazy import lazy_getattr

    with pytest.raises(AttributeError):
        lazy_getattr(
            name="nonexistent",
            module_name="enn",
            package="enn",
            mapping={},
            extra="pip install ennbo[with-deps]",
        )


# ---------------------------------------------------------------------------
# Scripts
# ---------------------------------------------------------------------------


def test_bench_result():
    from scripts.bench_raasp_time import BenchResult

    r = BenchResult(num_candidates=100, times_s=[0.1, 0.2])
    assert r.num_candidates == 100
    assert r.error is None


def test_bench_raasp():
    from scripts.bench_raasp_time import bench_raasp

    results = bench_raasp(num_dim=3, num_candidates_list=[10], repeats=1, seed=42)
    assert len(results) == 1
    assert results[0].error is None


def test_bench_raasp_main(monkeypatch):
    from scripts.bench_raasp_time import main

    monkeypatch.setattr(
        "sys.argv",
        [
            "bench_raasp_time",
            "--num-dim",
            "3",
            "--candidates",
            "10",
            "--repeats",
            "1",
            "--seed",
            "0",
        ],
    )
    main()


def test_benchmark_d_scaling(monkeypatch):
    from scripts.bench_d_scaling import benchmark_d_scaling

    benchmark_d_scaling(ds=[10, 20], n=10, num_candidates=20)


def test_profile_config():
    from scripts.profile_turbo_enn import ProfileConfig

    cfg = ProfileConfig(
        num_dim=2,
        num_obs=10,
        num_arms=2,
        num_candidates=20,
        num_fit_samples=5,
        num_fit_candidates=10,
        seed=0,
    )
    assert cfg.num_dim == 2


def test_run_profile():
    from scripts.profile_turbo_enn import ProfileConfig, run_profile

    cfg = ProfileConfig(
        num_dim=2,
        num_obs=10,
        num_arms=2,
        num_candidates=50,
        num_fit_samples=5,
        num_fit_candidates=10,
        seed=0,
    )
    run_profile(cfg, profile=False, profile_center=False)


def test_run_sweep():
    from scripts.profile_turbo_enn import ProfileConfig, run_sweep

    cfg = ProfileConfig(
        num_dim=2,
        num_obs=10,
        num_arms=2,
        num_candidates=50,
        num_fit_samples=5,
        num_fit_candidates=10,
        seed=0,
    )
    run_sweep(cfg, num_obs_values=[5, 10])


def test_profile_main(monkeypatch):
    from scripts.profile_turbo_enn import main

    monkeypatch.setattr(
        "sys.argv",
        [
            "profile_turbo_enn",
            "--num-dim",
            "2",
            "--num-obs",
            "10",
            "--num-arms",
            "2",
            "--num-candidates",
            "50",
            "--num-fit-samples",
            "5",
            "--num-fit_candidates",
            "10",
            "--seed",
            "0",
        ],
    )
    main()
