from __future__ import annotations

from dataclasses import dataclass

from .config.optimizer_config import OptimizerConfig
from .config.surrogate import GPSurrogateConfig


@dataclass(frozen=True)
class FallbackEntry:
    id: str
    reason: str
    owner: str
    expiration: str


FALLBACK_REGISTRY: tuple[FallbackEntry, ...] = (
    FallbackEntry(
        id="gpsurrogate_turbo_one",
        reason="GP is GPyTorch-only in first milestone",
        owner="gp_port",
        expiration="phase_3",
    ),
)


def requires_python_optimizer_fallback(config: OptimizerConfig) -> bool:
    return isinstance(config.surrogate, GPSurrogateConfig)


def fallback_reason(config: OptimizerConfig) -> str | None:
    if isinstance(config.surrogate, GPSurrogateConfig):
        return "gpsurrogate_turbo_one"
    return None
