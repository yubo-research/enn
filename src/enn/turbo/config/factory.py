from __future__ import annotations
from . import acquisition as acq
from . import surrogate as sur
from . import trust_region as tr
from .candidate_gen_config import CandidateGenConfig
from .num_candidates_fn import NumCandidatesFn, const_num_candidates
from .acq_type import AcqType
from .candidate_rv import CandidateRV
from .init_config import InitConfig
from .optimizer_config import ObservationHistoryConfig, OptimizerConfig


def _make_candidate_gen_config(
    candidate_rv: CandidateRV,
    num_candidates: NumCandidatesFn | int | None,
) -> CandidateGenConfig:
    if num_candidates is None:
        return CandidateGenConfig(candidate_rv=candidate_rv)
    if isinstance(num_candidates, int):
        num_candidates = const_num_candidates(num_candidates)
    return CandidateGenConfig(candidate_rv=candidate_rv, num_candidates=num_candidates)


def turbo_one_config(
    *,
    num_candidates: int | None = None,
    num_init: int | None = None,
    trailing_obs: int | None = None,
    trust_region: tr.TrustRegionConfig | None = None,
    candidate_rv: CandidateRV = CandidateRV.SOBOL,
) -> OptimizerConfig:
    return OptimizerConfig(
        trust_region=trust_region or tr.TurboTRConfig(),
        candidates=_make_candidate_gen_config(candidate_rv, num_candidates),
        init=InitConfig(num_init=num_init),
        surrogate=sur.GPSurrogateConfig(),
        acquisition=acq.DrawAcquisitionConfig(),
        acq_optimizer=acq.RAASPOptimizerConfig(),
        observation_history=ObservationHistoryConfig(trailing_obs=trailing_obs),
    )


def turbo_zero_config(
    *,
    num_candidates: int | None = None,
    num_init: int | None = None,
    trailing_obs: int | None = None,
    trust_region: tr.TrustRegionConfig | None = None,
    candidate_rv: CandidateRV = CandidateRV.SOBOL,
) -> OptimizerConfig:
    return OptimizerConfig(
        trust_region=trust_region or tr.TurboTRConfig(),
        candidates=_make_candidate_gen_config(candidate_rv, num_candidates),
        init=InitConfig(num_init=num_init),
        surrogate=sur.NoSurrogateConfig(),
        acquisition=acq.RandomAcquisitionConfig(),
        acq_optimizer=acq.RAASPOptimizerConfig(),
        observation_history=ObservationHistoryConfig(trailing_obs=trailing_obs),
    )


def turbo_enn_config(
    *,
    enn: sur.ENNSurrogateConfig | None = None,
    trust_region: tr.TrustRegionConfig | None = None,
    candidates: CandidateGenConfig | None = None,
    num_init: int | None = None,
    trailing_obs: int | None = None,
    acq_type: AcqType = AcqType.PARETO,
) -> OptimizerConfig:
    if acq_type == AcqType.PARETO:
        acquisition = acq.ParetoAcquisitionConfig()
        acq_optimizer = acq.NDSOptimizerConfig()
    elif acq_type == AcqType.UCB:
        acquisition = acq.UCBAcquisitionConfig()
        acq_optimizer = acq.RAASPOptimizerConfig()
    elif acq_type == AcqType.THOMPSON:
        acquisition = acq.DrawAcquisitionConfig()
        acq_optimizer = acq.RAASPOptimizerConfig()
    else:
        raise ValueError(
            f"acq_type must be AcqType.THOMPSON, AcqType.PARETO, or AcqType.UCB, got {acq_type!r}"
        )
    surrogate = enn if enn is not None else sur.ENNSurrogateConfig()
    if surrogate.num_fit_samples is None and acq_type != AcqType.PARETO:
        raise ValueError(f"enn.num_fit_samples required for acq_type={acq_type!r}")
    return OptimizerConfig(
        trust_region=trust_region or tr.TurboTRConfig(),
        candidates=candidates or CandidateGenConfig(),
        init=InitConfig(num_init=num_init),
        surrogate=surrogate,
        acquisition=acquisition,
        acq_optimizer=acq_optimizer,
        observation_history=ObservationHistoryConfig(trailing_obs=trailing_obs),
    )


def lhd_only_config(
    *,
    num_candidates: int | None = None,
    num_init: int | None = None,
    trailing_obs: int | None = None,
    trust_region: tr.TrustRegionConfig | None = None,
    candidate_rv: CandidateRV = CandidateRV.SOBOL,
) -> OptimizerConfig:
    from .init_strategies import LHDOnlyInit

    return OptimizerConfig(
        trust_region=trust_region or tr.NoTRConfig(),
        candidates=_make_candidate_gen_config(candidate_rv, num_candidates),
        init=InitConfig(init_strategy=LHDOnlyInit(), num_init=num_init),
        surrogate=sur.NoSurrogateConfig(),
        acquisition=acq.RandomAcquisitionConfig(),
        acq_optimizer=acq.RAASPOptimizerConfig(),
        observation_history=ObservationHistoryConfig(trailing_obs=trailing_obs),
    )
