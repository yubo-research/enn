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
from .enums import (
    AcqType,
    CandidateRV,
)
from .init_strategies import HybridInit, InitStrategy, LHDOnlyInit
from .surrogate import (
    ENNSurrogateConfig,
    GPSurrogateConfig,
    NoSurrogateConfig,
    SurrogateConfig,
)
from .trust_region import (
    MorboTRConfig,
    NoTRConfig,
    TrustRegionConfig,
    TurboTRConfig,
)
from .optimizer_config import OptimizerConfig


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
    "DrawAcquisitionConfig",
    "ENNSurrogateConfig",
    "GPSurrogateConfig",
    "HnROptimizerConfig",
    "InitConfig",
    "InitStrategy",
    "HybridInit",
    "LHDOnlyInit",
    "lhd_only_config",
    "MorboTRConfig",
    "NDSOptimizerConfig",
    "NoSurrogateConfig",
    "NoTRConfig",
    "OptimizerConfig",
    "ParetoAcquisitionConfig",
    "RAASPOptimizerConfig",
    "RandomAcquisitionConfig",
    "SurrogateConfig",
    "TrustRegionConfig",
    "turbo_enn_config",
    "turbo_one_config",
    "TurboTRConfig",
    "turbo_zero_config",
    "UCBAcquisitionConfig",
]
