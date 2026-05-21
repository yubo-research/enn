from .acq_type import AcqType
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
from .base import (
    CandidateGenConfig,
    InitConfig,
)
from .candidate_rv import CandidateRV
from .enn_index_driver import ENNIndexDriver
from .num_candidates_fn import default_num_candidates
from .optimizer_config import OptimizerConfig
from .raasp_driver import RAASPDriver
from .rescalarize import Rescalarize
from .surrogate import (
    ENNFitConfig,
    ENNSurrogateConfig,
    GPSurrogateConfig,
    NoSurrogateConfig,
    SurrogateConfig,
)
from .trust_region import (
    InitStrategy,
    MorboTRConfig,
    MultiObjectiveConfig,
    NoTRConfig,
    RescalePolicyConfig,
    TRLengthConfig,
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
    if name == "HybridInit":
        from .init_strategies.hybrid_init import HybridInit

        return HybridInit
    if name == "LHDOnlyInit":
        from .init_strategies.lhd_only_init import LHDOnlyInit

        return LHDOnlyInit
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AcqOptimizerConfig",
    "AcqType",
    "AcquisitionConfig",
    "CandidateGenConfig",
    "CandidateRV",
    "default_num_candidates",
    "ENNIndexDriver",
    "RAASPDriver",
    "Rescalarize",
    "DrawAcquisitionConfig",
    "ENNFitConfig",
    "ENNSurrogateConfig",
    "GPSurrogateConfig",
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
    "turbo_enn_config",
    "turbo_one_config",
    "TurboTRConfig",
    "turbo_zero_config",
    "UCBAcquisitionConfig",
]
