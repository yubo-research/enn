from __future__ import annotations

from . import acquisition as acq
from . import surrogate as sur
from . import trust_region as tr
from .acq_type import AcqType
from .candidate_gen_config import CandidateGenConfig
from .candidate_rv import CandidateRV
from .init_config import InitConfig
from .optimizer_config import ObservationHistoryConfig, OptimizerConfig


def _lhd_candidates(candidate_rv: CandidateRV | None) -> CandidateGenConfig:
    return CandidateGenConfig(
        candidate_rv=candidate_rv,
        min_candidates=1,
        max_candidates=1_000_000_000,
        num_candidates_per_dim=1,
        num_candidates_per_arm=0,
    )


def _acquisition(acq_type: AcqType) -> acq.AcquisitionConfig:
    if acq_type == AcqType.PARETO:
        return acq.ParetoAcquisitionConfig()
    if acq_type == AcqType.UCB:
        return acq.UCBAcquisitionConfig()
    if acq_type == AcqType.THOMPSON:
        return acq.DrawAcquisitionConfig()
    raise ValueError(
        f"acq_type must be AcqType.THOMPSON, AcqType.PARETO, or AcqType.UCB, got {acq_type!r}"
    )


def turbo_zero_config(
    *,
    candidates: CandidateGenConfig | None = None,
    num_init: int | None = None,
    trust_region: tr.TrustRegionConfig | None = None,
    candidate_rv: CandidateRV | None = None,
) -> OptimizerConfig:
    return OptimizerConfig(
        trust_region=trust_region or tr.TurboTRConfig(),
        candidates=candidates or CandidateGenConfig(candidate_rv=candidate_rv),
        init=InitConfig(num_init=num_init),
        surrogate=sur.NoSurrogateConfig(),
        acquisition=acq.RandomAcquisitionConfig(),
        observation_history=ObservationHistoryConfig(),
    )


def turbo_enn_config(
    *,
    enn: sur.ENNSurrogateConfig | None = None,
    trust_region: tr.TrustRegionConfig | None = None,
    candidates: CandidateGenConfig | None = None,
    num_init: int | None = None,
    acq_type: AcqType = AcqType.PARETO,
) -> OptimizerConfig:
    acquisition = _acquisition(acq_type)
    surrogate = enn if enn is not None else sur.ENNSurrogateConfig()
    from enn._rust import require_num_fit_samples

    require_num_fit_samples(acq.acquisition_kind(acquisition), surrogate.num_fit_samples)
    return OptimizerConfig(
        trust_region=trust_region or tr.TurboTRConfig(),
        candidates=candidates or CandidateGenConfig(),
        init=InitConfig(num_init=num_init),
        surrogate=surrogate,
        acquisition=acquisition,
        observation_history=ObservationHistoryConfig(),
    )


def lhd_only_config(
    *,
    candidates: CandidateGenConfig | None = None,
    num_init: int | None = None,
    trust_region: tr.TrustRegionConfig | None = None,
    candidate_rv: CandidateRV | None = None,
) -> OptimizerConfig:
    from .init_strategies import LHDOnlyInit

    return OptimizerConfig(
        trust_region=trust_region or tr.NoTRConfig(),
        candidates=candidates or _lhd_candidates(candidate_rv),
        init=InitConfig(init_strategy=LHDOnlyInit(), num_init=num_init),
        surrogate=sur.NoSurrogateConfig(),
        acquisition=acq.RandomAcquisitionConfig(),
        observation_history=ObservationHistoryConfig(),
    )
