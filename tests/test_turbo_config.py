from __future__ import annotations

import pytest

from enn.turbo.config import Rescalarize
from enn.turbo.config.turbo_tr_config import TRLengthConfig
from enn.turbo.config import (
    AcqType,
    CandidateGenConfig,
    CandidateRV,
    DrawAcquisitionConfig,
    ENNFitConfig,
    ENNSurrogateConfig,
    GPSurrogateConfig,
    HnROptimizerConfig,
    HybridInit,
    InitConfig,
    LHDOnlyInit,
    MorboTRConfig,
    MultiObjectiveConfig,
    NDSOptimizerConfig,
    NoSurrogateConfig,
    NoTRConfig,
    OptimizerConfig,
    ParetoAcquisitionConfig,
    RAASPOptimizerConfig,
    RandomAcquisitionConfig,
    RescalePolicyConfig,
    TurboTRConfig,
    UCBAcquisitionConfig,
    lhd_only_config,
    turbo_enn_config,
    turbo_one_config,
    turbo_zero_config,
)


def test_turbo_tr_config_defaults():
    cfg = TurboTRConfig()
    assert cfg.length_init == 0.8
    assert cfg.length_min == 0.5**7
    assert cfg.length_max == 1.6


def test_turbo_tr_config_custom():
    cfg = TurboTRConfig(
        length=TRLengthConfig(length_init=0.5, length_min=0.01, length_max=2.0)
    )
    assert cfg.length_init == 0.5
    assert cfg.length_min == 0.01
    assert cfg.length_max == 2.0


def test_turbo_tr_config_invalid():
    with pytest.raises(ValueError, match="length_init must be > 0"):
        TRLengthConfig(length_init=0)
    with pytest.raises(ValueError, match="length_min must be < length_max"):
        TRLengthConfig(length_min=1.0, length_max=0.5)


def test_turbo_tr_config_invalid_length_init_exceeds_max():
    with pytest.raises(ValueError, match="length_init must be <= length_max"):
        TRLengthConfig(length_init=2.0, length_max=1.0)


def test_turbo_tr_config_invalid_length_min_exceeds_init():
    with pytest.raises(ValueError, match="length_min must be <= length_init"):
        TRLengthConfig(length_init=0.05, length_min=0.1)


def test_morbo_tr_config():
    cfg = MorboTRConfig(multi_objective=MultiObjectiveConfig(num_metrics=3))
    assert cfg.num_metrics == 3
    assert cfg.alpha == 0.05


def test_morbo_tr_config_invalid():
    with pytest.raises(ValueError, match="num_metrics must be >= 2"):
        MorboTRConfig(multi_objective=MultiObjectiveConfig(num_metrics=1))


def test_morbo_tr_config_custom_alpha():
    cfg = MorboTRConfig(multi_objective=MultiObjectiveConfig(num_metrics=2, alpha=0.1))
    assert cfg.alpha == 0.1


def test_morbo_tr_config_invalid_alpha():
    with pytest.raises(ValueError, match="alpha must be > 0"):
        MorboTRConfig(multi_objective=MultiObjectiveConfig(num_metrics=2, alpha=0))


def test_multi_objective_config_defaults():
    cfg = MultiObjectiveConfig(num_metrics=2)
    assert cfg.num_metrics == 2
    assert cfg.alpha == 0.05


def test_multi_objective_config_custom():
    cfg = MultiObjectiveConfig(num_metrics=3, alpha=0.1)
    assert cfg.num_metrics == 3
    assert cfg.alpha == 0.1


def test_morbo_tr_config_length_defaults():
    cfg = MorboTRConfig(multi_objective=MultiObjectiveConfig(num_metrics=2))
    assert cfg.length_init == 0.8
    assert cfg.length_min == 0.5**7
    assert cfg.length_max == 1.6


def test_morbo_tr_config_custom_length():
    cfg = MorboTRConfig(
        multi_objective=MultiObjectiveConfig(num_metrics=2),
        length=TRLengthConfig(
            length_init=0.5,
            length_min=0.01,
            length_max=2.0,
        ),
    )
    assert cfg.length_init == 0.5
    assert cfg.length_min == 0.01
    assert cfg.length_max == 2.0


def test_morbo_tr_config_rescalarize_default():
    cfg = MorboTRConfig(multi_objective=MultiObjectiveConfig(num_metrics=2))
    assert cfg.rescalarize == Rescalarize.ON_PROPOSE


def test_morbo_tr_config_rescalarize_custom():
    cfg = MorboTRConfig(
        multi_objective=MultiObjectiveConfig(num_metrics=2),
        rescale_policy=RescalePolicyConfig(rescalarize=Rescalarize.ON_RESTART),
    )
    assert cfg.rescalarize == Rescalarize.ON_RESTART


def test_rescale_policy_config_default():
    cfg = RescalePolicyConfig()
    assert cfg.rescalarize == Rescalarize.ON_PROPOSE


def test_rescale_policy_config_custom():
    cfg = RescalePolicyConfig(rescalarize=Rescalarize.ON_RESTART)
    assert cfg.rescalarize == Rescalarize.ON_RESTART


def test_no_tr_config():
    cfg = NoTRConfig()
    assert cfg is not None


def test_candidate_gen_config_defaults():
    cfg = CandidateGenConfig()
    assert cfg.candidate_rv == CandidateRV.SOBOL
    assert callable(cfg.num_candidates)
    assert cfg.num_candidates(num_dim=1, num_arms=1) == 100
    assert cfg.num_candidates(num_dim=100, num_arms=1) == 5000


def test_candidate_gen_config_uniform():
    cfg = CandidateGenConfig(
        candidate_rv=CandidateRV.UNIFORM,
        num_candidates=lambda *, num_dim, num_arms: 100,
    )
    assert cfg.candidate_rv == CandidateRV.UNIFORM
    assert callable(cfg.num_candidates)
    assert cfg.num_candidates(num_dim=3, num_arms=7) == 100


def test_candidate_gen_config_invalid_rv():
    with pytest.raises(ValueError, match="candidate_rv must be"):
        CandidateGenConfig(candidate_rv="invalid")


def test_candidate_gen_config_invalid_num_candidates():
    with pytest.raises(ValueError, match="num_candidates must be > 0"):
        CandidateGenConfig(num_candidates=lambda *, num_dim, num_arms: 0)


def test_candidate_gen_config_num_candidates_per_arms():
    cfg = CandidateGenConfig(num_candidates=lambda *, num_dim, num_arms: 100 * num_arms)
    assert cfg.num_candidates(num_dim=3, num_arms=7) == 700


def test_init_config_defaults():
    cfg = InitConfig()
    assert isinstance(cfg.init_strategy, HybridInit)
    assert cfg.num_init is None


def test_init_config_lhd_only():
    cfg = InitConfig(init_strategy=LHDOnlyInit(), num_init=20)
    assert isinstance(cfg.init_strategy, LHDOnlyInit)
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
    assert cfg.num_fit_candidates is None
    assert cfg.scale_x is False


def test_enn_surrogate_config_with_values():
    cfg = ENNSurrogateConfig(k=10, fit=ENNFitConfig(num_fit_samples=50), scale_x=True)
    assert cfg.k == 10
    assert cfg.num_fit_samples == 50
    assert cfg.scale_x is True


def test_enn_fit_config_defaults():
    cfg = ENNFitConfig()
    assert cfg.num_fit_samples is None
    assert cfg.num_fit_candidates is None


def test_enn_fit_config_custom():
    cfg = ENNFitConfig(num_fit_samples=50, num_fit_candidates=100)
    assert cfg.num_fit_samples == 50
    assert cfg.num_fit_candidates == 100


def test_enn_fit_config_invalid_num_fit_samples():
    with pytest.raises(ValueError, match="num_fit_samples must be > 0"):
        ENNFitConfig(num_fit_samples=0)


def test_enn_fit_config_invalid_num_fit_candidates():
    with pytest.raises(ValueError, match="num_fit_candidates must be > 0"):
        ENNFitConfig(num_fit_candidates=0)


def test_enn_surrogate_config_num_fit_candidates_custom():
    cfg = ENNSurrogateConfig(fit=ENNFitConfig(num_fit_candidates=100))
    assert cfg.num_fit_candidates == 100


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


def test_hnr_optimizer_config():
    cfg = HnROptimizerConfig()
    assert cfg is not None


def test_optimizer_config_defaults():
    cfg = OptimizerConfig()
    assert isinstance(cfg.trust_region, TurboTRConfig)
    assert cfg.candidate_rv == CandidateRV.SOBOL
    assert cfg.init.num_init is None
    assert isinstance(cfg.surrogate, NoSurrogateConfig)


def test_optimizer_config_lhd_only_requires_no_surrogate():
    with pytest.raises(
        ValueError, match="init_strategy='lhd_only' requires NoSurrogateConfig"
    ):
        OptimizerConfig(
            init=InitConfig(init_strategy=LHDOnlyInit()),
            surrogate=GPSurrogateConfig(),
        )
    config = OptimizerConfig(
        init=InitConfig(init_strategy=LHDOnlyInit()),
        trust_region=MorboTRConfig(multi_objective=MultiObjectiveConfig(num_metrics=2)),
        surrogate=NoSurrogateConfig(),
    )
    assert config.num_metrics == 2


def test_optimizer_config_no_surrogate_rejects_draw():
    with pytest.raises(
        ValueError,
        match="NoSurrogateConfig is not compatible with DrawAcquisitionConfig",
    ):
        OptimizerConfig(
            surrogate=NoSurrogateConfig(),
            acquisition=DrawAcquisitionConfig(),
        )


def test_optimizer_config_no_surrogate_rejects_ucb():
    with pytest.raises(
        ValueError,
        match="NoSurrogateConfig is not compatible with UCBAcquisitionConfig",
    ):
        OptimizerConfig(
            surrogate=NoSurrogateConfig(),
            acquisition=UCBAcquisitionConfig(),
        )


def test_optimizer_config_pareto_requires_nds():
    with pytest.raises(
        ValueError, match="ParetoAcquisitionConfig requires NDSOptimizerConfig"
    ):
        OptimizerConfig(
            acquisition=ParetoAcquisitionConfig(),
            acq_optimizer=RAASPOptimizerConfig(),
        )


def test_optimizer_config_hnr_incompatible_with_pareto():
    with pytest.raises(
        ValueError, match="ParetoAcquisitionConfig requires NDSOptimizerConfig"
    ):
        OptimizerConfig(
            acquisition=ParetoAcquisitionConfig(),
            acq_optimizer=HnROptimizerConfig(),
        )


def test_optimizer_config_gp_draw_hnr_nyi():
    with pytest.raises(NotImplementedError, match="not yet implemented"):
        OptimizerConfig(
            surrogate=GPSurrogateConfig(),
            acquisition=DrawAcquisitionConfig(),
            acq_optimizer=HnROptimizerConfig(),
        )


def test_optimizer_config_enn_ucb_hnr_valid():
    cfg = OptimizerConfig(
        surrogate=ENNSurrogateConfig(fit=ENNFitConfig(num_fit_samples=50)),
        acquisition=UCBAcquisitionConfig(),
        acq_optimizer=HnROptimizerConfig(),
    )
    assert isinstance(cfg.acq_optimizer, HnROptimizerConfig)
    assert isinstance(cfg.acquisition, UCBAcquisitionConfig)


def test_optimizer_config_enn_draw_hnr_valid():
    cfg = OptimizerConfig(
        surrogate=ENNSurrogateConfig(fit=ENNFitConfig(num_fit_samples=50)),
        acquisition=DrawAcquisitionConfig(),
        acq_optimizer=HnROptimizerConfig(),
    )
    assert isinstance(cfg.acq_optimizer, HnROptimizerConfig)
    assert isinstance(cfg.acquisition, DrawAcquisitionConfig)


def test_optimizer_config_gp_ucb_hnr_valid():
    cfg = OptimizerConfig(
        surrogate=GPSurrogateConfig(),
        acquisition=UCBAcquisitionConfig(),
        acq_optimizer=HnROptimizerConfig(),
    )
    assert isinstance(cfg.acq_optimizer, HnROptimizerConfig)
    assert isinstance(cfg.acquisition, UCBAcquisitionConfig)


def test_turbo_one_config_factory():
    cfg = turbo_one_config()
    assert isinstance(cfg.trust_region, TurboTRConfig)
    assert isinstance(cfg.surrogate, GPSurrogateConfig)
    assert isinstance(cfg.acquisition, DrawAcquisitionConfig)


def test_turbo_zero_config_factory():
    cfg = turbo_zero_config()
    assert isinstance(cfg.trust_region, TurboTRConfig)
    assert isinstance(cfg.surrogate, NoSurrogateConfig)
    assert isinstance(cfg.acquisition, RandomAcquisitionConfig)


def test_turbo_enn_config_factory_pareto():
    cfg = turbo_enn_config(acq_type=AcqType.PARETO)
    assert isinstance(cfg.surrogate, ENNSurrogateConfig)
    assert isinstance(cfg.acquisition, ParetoAcquisitionConfig)
    assert isinstance(cfg.acq_optimizer, NDSOptimizerConfig)


def test_turbo_enn_config_factory_ucb():
    cfg = turbo_enn_config(
        acq_type=AcqType.UCB,
        enn=ENNSurrogateConfig(fit=ENNFitConfig(num_fit_samples=50)),
    )
    assert isinstance(cfg.surrogate, ENNSurrogateConfig)
    assert isinstance(cfg.acquisition, UCBAcquisitionConfig)
    assert isinstance(cfg.acq_optimizer, RAASPOptimizerConfig)


def test_turbo_enn_config_factory_thompson():
    cfg = turbo_enn_config(
        acq_type=AcqType.THOMPSON,
        enn=ENNSurrogateConfig(fit=ENNFitConfig(num_fit_samples=50)),
    )
    assert isinstance(cfg.surrogate, ENNSurrogateConfig)
    assert isinstance(cfg.acquisition, DrawAcquisitionConfig)


def test_turbo_enn_config_requires_num_fit_samples_for_non_pareto():
    with pytest.raises(ValueError, match="num_fit_samples required"):
        turbo_enn_config(acq_type=AcqType.UCB)


def test_lhd_only_config_factory():
    cfg = lhd_only_config()
    assert isinstance(cfg.trust_region, NoTRConfig)
    assert isinstance(cfg.init.init_strategy, LHDOnlyInit)
    assert isinstance(cfg.surrogate, NoSurrogateConfig)


def test_optimizer_config_properties():
    cfg = turbo_enn_config(
        enn=ENNSurrogateConfig(k=15),
        candidates=CandidateGenConfig(num_candidates=lambda *, num_dim, num_arms: 200),
        num_init=10,
    )
    assert cfg.surrogate.k == 15
    assert cfg.num_candidates(num_dim=3, num_arms=7) == 200
    assert cfg.init.num_init == 10


def test_optimizer_config_num_metrics():
    cfg_turbo = OptimizerConfig(trust_region=TurboTRConfig())
    assert cfg_turbo.num_metrics is None
    cfg_morbo = OptimizerConfig(
        trust_region=MorboTRConfig(multi_objective=MultiObjectiveConfig(num_metrics=2))
    )
    assert cfg_morbo.num_metrics == 2
    cfg_none = OptimizerConfig(trust_region=NoTRConfig())
    assert cfg_none.num_metrics is None
