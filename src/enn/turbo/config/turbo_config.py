from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .acquisition import (
    AcqOptimizerConfig,
    AcquisitionConfig,
    DrawAcquisitionConfig,
    NDSOptimizerConfig,
    ParetoAcquisitionConfig,
    RAASPOptimizerConfig,
    RandomAcquisitionConfig,
    UCBAcquisitionConfig,
)
from .base import CandidateGenConfig, InitConfig, TrustRegionConfig
from .surrogate import (
    ENNSurrogateConfig,
    GPSurrogateConfig,
    NoSurrogateConfig,
    SurrogateConfig,
)


def _validate_turbo_config(cfg: TurboConfig) -> None:
    if cfg.init.init_strategy == "lhd_only":
        if not isinstance(cfg.surrogate, NoSurrogateConfig):
            raise ValueError(
                "init_strategy='lhd_only' requires NoSurrogateConfig surrogate"
            )
        if cfg.trust_region.tr_type != "none":
            raise ValueError(
                f"init_strategy='lhd_only' requires tr_type='none', "
                f"got {cfg.trust_region.tr_type!r}"
            )

    if isinstance(cfg.acquisition, ParetoAcquisitionConfig):
        if not isinstance(cfg.acq_optimizer, NDSOptimizerConfig):
            raise ValueError("ParetoAcquisitionConfig requires NDSOptimizerConfig")


@dataclass(frozen=True)
class TurboConfig:
    trust_region: TrustRegionConfig = TrustRegionConfig()
    candidates: CandidateGenConfig = CandidateGenConfig()
    init: InitConfig = InitConfig()
    surrogate: SurrogateConfig = NoSurrogateConfig()
    acquisition: AcquisitionConfig = RandomAcquisitionConfig()
    acq_optimizer: AcqOptimizerConfig = RAASPOptimizerConfig()

    trailing_obs: int | None = None

    def __post_init__(self) -> None:
        _validate_turbo_config(self)

    @property
    def tr_type(self) -> Literal["turbo", "morbo", "none"]:
        return self.trust_region.tr_type

    @property
    def num_metrics(self) -> int | None:
        return self.trust_region.num_metrics

    @property
    def candidate_rv(self) -> Literal["sobol", "uniform"]:
        return self.candidates.candidate_rv

    @property
    def num_candidates(self) -> int | None:
        return self.candidates.num_candidates

    @property
    def num_init(self) -> int | None:
        return self.init.num_init

    @property
    def k(self) -> int | None:
        if isinstance(self.surrogate, ENNSurrogateConfig):
            return self.surrogate.k
        return None


def TurboOneConfig(
    *,
    k: int | None = None,
    num_candidates: int | None = None,
    num_init: int | None = None,
    trailing_obs: int | None = None,
    tr_type: Literal["turbo", "morbo", "none"] = "turbo",
    num_metrics: int | None = None,
    candidate_rv: Literal["sobol", "uniform"] = "sobol",
) -> TurboConfig:
    return TurboConfig(
        trust_region=TrustRegionConfig(tr_type=tr_type, num_metrics=num_metrics),
        candidates=CandidateGenConfig(
            candidate_rv=candidate_rv, num_candidates=num_candidates
        ),
        init=InitConfig(init_strategy="hybrid", num_init=num_init),
        surrogate=GPSurrogateConfig(),
        acquisition=DrawAcquisitionConfig(),
        acq_optimizer=RAASPOptimizerConfig(),
        trailing_obs=trailing_obs,
    )


def TurboZeroConfig(
    *,
    k: int | None = None,
    num_candidates: int | None = None,
    num_init: int | None = None,
    trailing_obs: int | None = None,
    tr_type: Literal["turbo", "morbo", "none"] = "turbo",
    num_metrics: int | None = None,
    candidate_rv: Literal["sobol", "uniform"] = "sobol",
) -> TurboConfig:
    return TurboConfig(
        trust_region=TrustRegionConfig(tr_type=tr_type, num_metrics=num_metrics),
        candidates=CandidateGenConfig(
            candidate_rv=candidate_rv, num_candidates=num_candidates
        ),
        init=InitConfig(init_strategy="hybrid", num_init=num_init),
        surrogate=NoSurrogateConfig(),
        acquisition=RandomAcquisitionConfig(),
        acq_optimizer=RAASPOptimizerConfig(),
        trailing_obs=trailing_obs,
    )


def TurboENNConfig(
    *,
    enn: ENNSurrogateConfig | None = None,
    trust_region: TrustRegionConfig | None = None,
    candidates: CandidateGenConfig | None = None,
    num_init: int | None = None,
    trailing_obs: int | None = None,
    acq_type: Literal["thompson", "pareto", "ucb"] = "pareto",
    num_fit_samples: int | None = None,
) -> TurboConfig:
    if acq_type == "pareto":
        acquisition: AcquisitionConfig = ParetoAcquisitionConfig()
        acq_optimizer: AcqOptimizerConfig = NDSOptimizerConfig()
    elif acq_type == "ucb":
        acquisition = UCBAcquisitionConfig()
        acq_optimizer = RAASPOptimizerConfig()
    elif acq_type == "thompson":
        acquisition = DrawAcquisitionConfig()
        acq_optimizer = RAASPOptimizerConfig()
    else:
        raise ValueError(
            f"acq_type must be 'thompson', 'pareto', or 'ucb', got {acq_type!r}"
        )

    if num_fit_samples is None and acq_type != "pareto":
        raise ValueError(f"num_fit_samples required for acq_type={acq_type!r}")

    if enn is not None:
        surrogate = enn
    else:
        surrogate = ENNSurrogateConfig(num_fit_samples=num_fit_samples)

    return TurboConfig(
        trust_region=trust_region or TrustRegionConfig(),
        candidates=candidates or CandidateGenConfig(),
        init=InitConfig(init_strategy="hybrid", num_init=num_init),
        surrogate=surrogate,
        acquisition=acquisition,
        acq_optimizer=acq_optimizer,
        trailing_obs=trailing_obs,
    )


def LHDOnlyConfig(
    *,
    k: int | None = None,
    num_candidates: int | None = None,
    num_init: int | None = None,
    trailing_obs: int | None = None,
    num_metrics: int | None = None,
    candidate_rv: Literal["sobol", "uniform"] = "sobol",
) -> TurboConfig:
    return TurboConfig(
        trust_region=TrustRegionConfig(tr_type="none", num_metrics=num_metrics),
        candidates=CandidateGenConfig(
            candidate_rv=candidate_rv, num_candidates=num_candidates
        ),
        init=InitConfig(init_strategy="lhd_only", num_init=num_init),
        surrogate=NoSurrogateConfig(),
        acquisition=RandomAcquisitionConfig(),
        acq_optimizer=RAASPOptimizerConfig(),
        trailing_obs=trailing_obs,
    )
