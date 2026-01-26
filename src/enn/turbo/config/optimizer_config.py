from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from .candidate_gen_config import CandidateGenConfig
from .init_config import InitConfig
from .surrogate import NoSurrogateConfig, SurrogateConfig
from .trust_region import TrustRegionConfig, TurboTRConfig
from .observation_history_config import ObservationHistoryConfig

if TYPE_CHECKING:
    from .acquisition import AcqOptimizerConfig, AcquisitionConfig
    from .candidate_rv import CandidateRV
    from .raasp_driver import RAASPDriver


def _default_acquisition():
    from .acquisition import RandomAcquisitionConfig

    return RandomAcquisitionConfig()


def _default_acq_optimizer():
    from .raasp_optimizer_config import RAASPOptimizerConfig

    return RAASPOptimizerConfig()


@dataclass(frozen=True)
class OptimizerConfig:
    trust_region: TrustRegionConfig = TurboTRConfig()
    candidates: CandidateGenConfig = CandidateGenConfig()
    init: InitConfig = InitConfig()
    surrogate: SurrogateConfig = NoSurrogateConfig()
    acquisition: AcquisitionConfig = field(default_factory=_default_acquisition)
    acq_optimizer: AcqOptimizerConfig = field(default_factory=_default_acq_optimizer)
    observation_history: ObservationHistoryConfig = ObservationHistoryConfig()

    def __post_init__(self) -> None:
        from .validation import validate_optimizer_config

        validate_optimizer_config(self)

    @property
    def num_metrics(self) -> int | None:
        from .morbo_tr_config import MorboTRConfig

        if isinstance(self.trust_region, MorboTRConfig):
            return self.trust_region.num_metrics
        return None

    @property
    def candidate_rv(self) -> CandidateRV:
        return self.candidates.candidate_rv

    @property
    def raasp_driver(self) -> RAASPDriver:
        return self.candidates.raasp_driver

    @property
    def num_candidates(self):
        return self.candidates.num_candidates

    @property
    def trailing_obs(self) -> int | None:
        return self.observation_history.trailing_obs
