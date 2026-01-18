from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .base import CandidateGenConfig, InitConfig
from .surrogate import ENNSurrogateConfig, NoSurrogateConfig, SurrogateConfig
from .trust_region import TrustRegionConfig, TurboTRConfig

if TYPE_CHECKING:
    from .acquisition import AcqOptimizerConfig, AcquisitionConfig
    from .enums import CandidateRV


@dataclass(frozen=True)
class ObservationHistoryConfig:
    trailing_obs: int | None = None

    def __post_init__(self) -> None:
        if self.trailing_obs is not None and self.trailing_obs <= 0:
            raise ValueError(f"trailing_obs must be > 0, got {self.trailing_obs}")


def _default_acquisition():
    from .acquisition import RandomAcquisitionConfig

    return RandomAcquisitionConfig()


def _default_acq_optimizer():
    from .acquisition import RAASPOptimizerConfig

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
        """Get num_metrics from MorboTRConfig if applicable."""
        from .morbo_tr_config import MorboTRConfig

        if isinstance(self.trust_region, MorboTRConfig):
            return self.trust_region.num_metrics
        return None

    @property
    def candidate_rv(self) -> CandidateRV:
        return self.candidates.candidate_rv

    @property
    def num_candidates(self):
        return self.candidates.num_candidates

    @property
    def num_init(self) -> int | None:
        return self.init.num_init

    @property
    def k(self) -> int | None:
        if isinstance(self.surrogate, ENNSurrogateConfig):
            return self.surrogate.k
        return None

    @property
    def trailing_obs(self) -> int | None:
        return self.observation_history.trailing_obs
