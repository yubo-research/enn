from .enn_surrogate_config import ENNSurrogateConfig
from .gp_surrogate_config import GPSurrogateConfig
from .no_surrogate_config import NoSurrogateConfig

SurrogateConfig = NoSurrogateConfig | GPSurrogateConfig | ENNSurrogateConfig

__all__ = [
    "ENNSurrogateConfig",
    "GPSurrogateConfig",
    "NoSurrogateConfig",
    "SurrogateConfig",
]
