from .enn_surrogate_config import ENNFitConfig, ENNSurrogateConfig
from .no_surrogate_config import NoSurrogateConfig

SurrogateConfig = NoSurrogateConfig | ENNSurrogateConfig
__all__ = [
    "ENNFitConfig",
    "ENNSurrogateConfig",
    "NoSurrogateConfig",
    "SurrogateConfig",
]
