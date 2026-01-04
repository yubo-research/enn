from __future__ import annotations

import pytest

from enn.turbo.turbo_config import (
    CandidateGenConfig,
    DrawAcquisitionConfig,
    ENNSurrogateConfig,
    GPSurrogateConfig,
    InitConfig,
    LHDOnlyConfig,
    NDSOptimizerConfig,
    NoSurrogateConfig,
    ParetoAcquisitionConfig,
    RAASPOptimizerConfig,
    RandomAcquisitionConfig,
    TrustRegionConfig,
    TurboConfig,
    TurboENNConfig,
    TurboOneConfig,
    TurboZeroConfig,
    UCBAcquisitionConfig,
)


def test_trust_region_config_defaults():
    cfg = TrustRegionConfig()
    assert cfg.tr_type == "turbo"
    assert cfg.num_metrics is None


def test_trust_region_config_morbo():
    cfg = TrustRegionConfig(tr_type="morbo", num_metrics=2)
    assert cfg.tr_type == "morbo"
    assert cfg.num_metrics == 2


def test_trust_region_config_invalid_tr_type():
    with pytest.raises(ValueError, match="tr_type must be"):
        TrustRegionConfig(tr_type="invalid")


def test_candidate_gen_config_defaults():
    cfg = CandidateGenConfig()
    assert cfg.candidate_rv == "sobol"
    assert cfg.num_candidates is None


def test_candidate_gen_config_uniform():
    cfg = CandidateGenConfig(candidate_rv="uniform", num_candidates=100)
    assert cfg.candidate_rv == "uniform"
    assert cfg.num_candidates == 100


def test_candidate_gen_config_invalid_rv():
    with pytest.raises(ValueError, match="candidate_rv must be"):
        CandidateGenConfig(candidate_rv="invalid")


def test_candidate_gen_config_invalid_num_candidates():
    with pytest.raises(ValueError, match="num_candidates must be > 0"):
        CandidateGenConfig(num_candidates=0)


def test_init_config_defaults():
    cfg = InitConfig()
    assert cfg.init_strategy == "hybrid"
    assert cfg.num_init is None


def test_init_config_lhd_only():
    cfg = InitConfig(init_strategy="lhd_only", num_init=20)
    assert cfg.init_strategy == "lhd_only"
    assert cfg.num_init == 20


def test_init_config_invalid_strategy():
    with pytest.raises(ValueError, match="init_strategy must be"):
        InitConfig(init_strategy="invalid")


def test_init_config_invalid_num_init():
    with pytest.raises(ValueError, match="num_init must be > 0"):
        InitConfig(num_init=0)


def test_no_surrogate_config():
    cfg = NoSurrogateConfig()
    assert cfg is not None


def test_gp_surrogate_config():
    cfg = GPSurrogateConfig()
    assert cfg is not None


def test_enn_surrogate_config_defaults():
    cfg = ENNSurrogateConfig()
    assert cfg.k is None
    assert cfg.num_fit_samples is None
    assert cfg.scale_x is False


def test_enn_surrogate_config_with_values():
    cfg = ENNSurrogateConfig(k=10, num_fit_samples=50, scale_x=True)
    assert cfg.k == 10
    assert cfg.num_fit_samples == 50
    assert cfg.scale_x is True


def test_enn_surrogate_config_invalid_num_fit_samples():
    with pytest.raises(ValueError, match="num_fit_samples must be > 0"):
        ENNSurrogateConfig(num_fit_samples=0)


def test_ucb_acquisition_config():
    cfg = UCBAcquisitionConfig()
    assert cfg is not None


def test_draw_acquisition_config():
    cfg = DrawAcquisitionConfig()
    assert cfg is not None


def test_pareto_acquisition_config():
    cfg = ParetoAcquisitionConfig()
    assert cfg is not None


def test_random_acquisition_config():
    cfg = RandomAcquisitionConfig()
    assert cfg is not None


def test_raasp_optimizer_config():
    cfg = RAASPOptimizerConfig()
    assert cfg is not None


def test_nds_optimizer_config():
    cfg = NDSOptimizerConfig()
    assert cfg is not None


def test_turbo_config_defaults():
    cfg = TurboConfig()
    assert cfg.tr_type == "turbo"
    assert cfg.candidate_rv == "sobol"
    assert cfg.num_init is None
    assert isinstance(cfg.surrogate, NoSurrogateConfig)


def test_turbo_config_lhd_only_requires_no_surrogate_and_none_tr():
    with pytest.raises(
        ValueError, match="init_strategy='lhd_only' requires NoSurrogateConfig"
    ):
        TurboConfig(
            init=InitConfig(init_strategy="lhd_only"),
            surrogate=GPSurrogateConfig(),
        )

    with pytest.raises(
        ValueError, match="init_strategy='lhd_only' requires tr_type='none'"
    ):
        TurboConfig(
            init=InitConfig(init_strategy="lhd_only"),
            trust_region=TrustRegionConfig(tr_type="turbo"),
            surrogate=NoSurrogateConfig(),
        )


def test_turbo_config_pareto_requires_nds():
    with pytest.raises(
        ValueError, match="ParetoAcquisitionConfig requires NDSOptimizerConfig"
    ):
        TurboConfig(
            acquisition=ParetoAcquisitionConfig(),
            acq_optimizer=RAASPOptimizerConfig(),
        )


def test_turbo_one_config_factory():
    cfg = TurboOneConfig()
    assert cfg.tr_type == "turbo"
    assert isinstance(cfg.surrogate, GPSurrogateConfig)
    assert isinstance(cfg.acquisition, DrawAcquisitionConfig)


def test_turbo_zero_config_factory():
    cfg = TurboZeroConfig()
    assert cfg.tr_type == "turbo"
    assert isinstance(cfg.surrogate, NoSurrogateConfig)
    assert isinstance(cfg.acquisition, RandomAcquisitionConfig)


def test_turbo_enn_config_factory_pareto():
    cfg = TurboENNConfig(acq_type="pareto")
    assert isinstance(cfg.surrogate, ENNSurrogateConfig)
    assert isinstance(cfg.acquisition, ParetoAcquisitionConfig)
    assert isinstance(cfg.acq_optimizer, NDSOptimizerConfig)


def test_turbo_enn_config_factory_ucb():
    cfg = TurboENNConfig(acq_type="ucb", num_fit_samples=50)
    assert isinstance(cfg.surrogate, ENNSurrogateConfig)
    assert isinstance(cfg.acquisition, UCBAcquisitionConfig)
    assert isinstance(cfg.acq_optimizer, RAASPOptimizerConfig)


def test_turbo_enn_config_factory_thompson():
    cfg = TurboENNConfig(acq_type="thompson", num_fit_samples=50)
    assert isinstance(cfg.surrogate, ENNSurrogateConfig)
    assert isinstance(cfg.acquisition, DrawAcquisitionConfig)


def test_turbo_enn_config_requires_num_fit_samples_for_non_pareto():
    with pytest.raises(ValueError, match="num_fit_samples required"):
        TurboENNConfig(acq_type="ucb")


def test_lhd_only_config_factory():
    cfg = LHDOnlyConfig()
    assert cfg.tr_type == "none"
    assert cfg.init.init_strategy == "lhd_only"
    assert isinstance(cfg.surrogate, NoSurrogateConfig)


def test_turbo_config_properties():
    cfg = TurboENNConfig(
        enn=ENNSurrogateConfig(k=15),
        candidates=CandidateGenConfig(num_candidates=200),
        num_init=10,
    )
    assert cfg.k == 15
    assert cfg.num_candidates == 200
    assert cfg.num_init == 10
