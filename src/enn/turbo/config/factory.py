from __future__ import annotations

from .acquisition import (
    DrawAcquisitionConfig,
    NDSOptimizerConfig,
    ParetoAcquisitionConfig,
    RAASPOptimizerConfig,
    RandomAcquisitionConfig,
    UCBAcquisitionConfig,
)
from .base import CandidateGenConfig, InitConfig
from .enums import AcqType, CandidateRV
from .init_strategies import HybridInit, LHDOnlyInit
from .optimizer_config import OptimizerConfig
from .surrogate import ENNSurrogateConfig, GPSurrogateConfig, NoSurrogateConfig
from .trust_region import NoTRConfig, TrustRegionConfig, TurboTRConfig


def _make_candidate_gen_config(
    candidate_rv: CandidateRV,
    num_candidates: int | None,
) -> CandidateGenConfig:
    """Create CandidateGenConfig, using default num_candidates if None."""
    if num_candidates is None:
        return CandidateGenConfig(candidate_rv=candidate_rv)
    return CandidateGenConfig(candidate_rv=candidate_rv, num_candidates=num_candidates)


def turbo_one_config(
    *,
    num_candidates: int | None = None,
    num_init: int | None = None,
    trailing_obs: int | None = None,
    trust_region: TrustRegionConfig | None = None,
    candidate_rv: CandidateRV = CandidateRV.SOBOL,
) -> OptimizerConfig:
    return OptimizerConfig(
        trust_region=trust_region or TurboTRConfig(),
        candidates=_make_candidate_gen_config(candidate_rv, num_candidates),
        init=InitConfig(init_strategy=HybridInit(), num_init=num_init),
        surrogate=GPSurrogateConfig(),
        acquisition=DrawAcquisitionConfig(),
        acq_optimizer=RAASPOptimizerConfig(),
        trailing_obs=trailing_obs,
    )


def turbo_zero_config(
    *,
    num_candidates: int | None = None,
    num_init: int | None = None,
    trailing_obs: int | None = None,
    trust_region: TrustRegionConfig | None = None,
    candidate_rv: CandidateRV = CandidateRV.SOBOL,
) -> OptimizerConfig:
    return OptimizerConfig(
        trust_region=trust_region or TurboTRConfig(),
        candidates=_make_candidate_gen_config(candidate_rv, num_candidates),
        init=InitConfig(init_strategy=HybridInit(), num_init=num_init),
        surrogate=NoSurrogateConfig(),
        acquisition=RandomAcquisitionConfig(),
        acq_optimizer=RAASPOptimizerConfig(),
        trailing_obs=trailing_obs,
    )


def turbo_enn_config(
    *,
    enn: ENNSurrogateConfig | None = None,
    trust_region: TrustRegionConfig | None = None,
    candidates: CandidateGenConfig | None = None,
    num_init: int | None = None,
    trailing_obs: int | None = None,
    acq_type: AcqType = AcqType.PARETO,
) -> OptimizerConfig:
    if acq_type == AcqType.PARETO:
        acquisition = ParetoAcquisitionConfig()
        acq_optimizer = NDSOptimizerConfig()
    elif acq_type == AcqType.UCB:
        acquisition = UCBAcquisitionConfig()
        acq_optimizer = RAASPOptimizerConfig()
    elif acq_type == AcqType.THOMPSON:
        acquisition = DrawAcquisitionConfig()
        acq_optimizer = RAASPOptimizerConfig()
    else:
        raise ValueError(
            f"acq_type must be AcqType.THOMPSON, AcqType.PARETO, or AcqType.UCB, got {acq_type!r}"
        )

    surrogate = enn if enn is not None else ENNSurrogateConfig()

    if surrogate.num_fit_samples is None and acq_type != AcqType.PARETO:
        raise ValueError(f"enn.num_fit_samples required for acq_type={acq_type!r}")

    return OptimizerConfig(
        trust_region=trust_region or TurboTRConfig(),
        candidates=candidates or CandidateGenConfig(),
        init=InitConfig(init_strategy=HybridInit(), num_init=num_init),
        surrogate=surrogate,
        acquisition=acquisition,
        acq_optimizer=acq_optimizer,
        trailing_obs=trailing_obs,
    )


def lhd_only_config(
    *,
    num_candidates: int | None = None,
    num_init: int | None = None,
    trailing_obs: int | None = None,
    trust_region: TrustRegionConfig | None = None,
    candidate_rv: CandidateRV = CandidateRV.SOBOL,
) -> OptimizerConfig:
    return OptimizerConfig(
        trust_region=trust_region or NoTRConfig(),
        candidates=_make_candidate_gen_config(candidate_rv, num_candidates),
        init=InitConfig(init_strategy=LHDOnlyInit(), num_init=num_init),
        surrogate=NoSurrogateConfig(),
        acquisition=RandomAcquisitionConfig(),
        acq_optimizer=RAASPOptimizerConfig(),
        trailing_obs=trailing_obs,
    )
