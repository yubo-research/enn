from .acquisition import (
    AcqOptimizerConfig,
    AcquisitionConfig,
    DrawAcquisitionConfig,
    HnROptimizerConfig,
    NDSOptimizerConfig,
    ParetoAcquisitionConfig,
    RAASPOptimizerConfig,
    RandomAcquisitionConfig,
    UCBAcquisitionConfig,
)
from .base import (
    CandidateGenConfig,
    InitConfig,
)
from .acq_type import AcqType
from .enn_index_driver import ENNIndexDriver
from .rescalarize import Rescalarize
from .candidate_rv import CandidateRV
from .raasp_driver import RAASPDriver
from .num_candidates_fn import (
    NumCandidatesFn,
    const_num_candidates,
    default_num_candidates,
)

__all__ = [
    "AcqType",
    "ENNIndexDriver",
    "Rescalarize",
    "CandidateRV",
    "RAASPDriver",
]
from .trust_region import InitStrategy
from .init_strategies.hybrid_init import HybridInit
from .init_strategies.lhd_only_init import LHDOnlyInit
from .optimizer_config import OptimizerConfig
from .surrogate import (
    ENNFitConfig,
    ENNSurrogateConfig,
    GPSurrogateConfig,
    NoSurrogateConfig,
    SurrogateConfig,
)
from .trust_region import (
    MorboTRConfig,
    MultiObjectiveConfig,
    NoTRConfig,
    RescalePolicyConfig,
    TRLengthConfig,
    TrustRegionConfig,
    TurboTRConfig,
)


def __getattr__(name: str) -> object:
    if name in (
        "lhd_only_config",
        "turbo_enn_config",
        "turbo_one_config",
        "turbo_zero_config",
    ):
        from . import factory

        return getattr(factory, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AcqOptimizerConfig",
    "AcqType",
    "AcquisitionConfig",
    "CandidateGenConfig",
    "CandidateRV",
    "const_num_candidates",
    "default_num_candidates",
    "NumCandidatesFn",
    "ENNIndexDriver",
    "RAASPDriver",
    "Rescalarize",
    "DrawAcquisitionConfig",
    "ENNFitConfig",
    "ENNSurrogateConfig",
    "GPSurrogateConfig",
    "HnROptimizerConfig",
    "InitConfig",
    "HybridInit",
    "InitStrategy",
    "LHDOnlyInit",
    "lhd_only_config",
    "MorboTRConfig",
    "MultiObjectiveConfig",
    "NDSOptimizerConfig",
    "NoSurrogateConfig",
    "NoTRConfig",
    "OptimizerConfig",
    "ParetoAcquisitionConfig",
    "RAASPOptimizerConfig",
    "RandomAcquisitionConfig",
    "RescalePolicyConfig",
    "SurrogateConfig",
    "TRLengthConfig",
    "TrustRegionConfig",
    "turbo_enn_config",
    "turbo_one_config",
    "TurboTRConfig",
    "turbo_zero_config",
    "UCBAcquisitionConfig",
]
