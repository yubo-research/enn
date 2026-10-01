from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Shared helpers (reduce duplication across config/trust-region tests)
# ---------------------------------------------------------------------------



def _load_tests_script(name: str):
    import importlib.util
    import sys
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def _morbo_tr_config():
    from enn.turbo.config import MorboTRConfig, MultiObjectiveConfig

    mo = MultiObjectiveConfig(num_metrics=2, alpha=0.05)
    return MorboTRConfig(multi_objective=mo)






def _enn_model():
    from enn.enn.enn_class import EpistemicNearestNeighbors

    x = np.array([[0.1, 0.2], [0.3, 0.4]])
    y = np.array([[1.0], [2.0]])
    return EpistemicNearestNeighbors(x, y)


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
    assert cfg.rescalarize == Rescalarize.ON_RESTART


def test_morbo_tr_config_properties():
    cfg = _morbo_tr_config()
    # num_metrics
    assert cfg.num_metrics == 2
    # alpha
    assert cfg.alpha == 0.05
    assert cfg.length_init is None
    assert cfg.length_min is None
    assert cfg.length_max is None


def test_turbo_tr_config_properties():
    from enn.turbo.config import TurboTRConfig

    cfg = TurboTRConfig()
    assert cfg.length_init is None
    assert cfg.length_min is None
    assert cfg.length_max is None


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

    assert ENNIndexDriver.FLAT != ENNIndexDriver.BPANN_DISK


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



# ---------------------------------------------------------------------------
# Component protocols / properties
# ---------------------------------------------------------------------------














# ---------------------------------------------------------------------------
# ENN class/index
# ---------------------------------------------------------------------------


def test_enn_class_properties():
    enn = _enn_model()
    _, y_at, yvar_at = enn.train_rows_at([0, 1])
    assert y_at.shape == (2, 1)
    assert yvar_at is None
    assert enn.num_outputs == 1


def test_enn_class_add():
    enn = _enn_model()
    enn.add(np.array([[0.5, 0.6]]), np.array([[3.0]]))
    assert len(enn) == 3


def test_enn_neighbor_distances_add_and_search():
    from enn.enn.enn_class_support import (
        enn_index_neighbor_distances_and_indices,
        enn_neighbor_distances_and_indices,
    )

    enn = _enn_model()
    enn.add(np.array([[0.5, 0.6]], dtype=float), np.array([[3.0]], dtype=float))
    _d2, nn = enn_neighbor_distances_and_indices(
        enn.rust_backend,
        np.array([[0.5, 0.6]], dtype=float),
        search_k=3,
        exclude_nearest=False,
    )
    assert nn.shape[1] == 3
    idx_d2, idx_nn = enn_index_neighbor_distances_and_indices(
        enn.rust_backend,
        np.array([[0.5, 0.6]], dtype=float),
        search_k=3,
        exclude_nearest=False,
    )
    assert idx_d2.shape == idx_nn.shape
    assert idx_d2.shape[1] == 3


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








# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------






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


