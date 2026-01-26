from __future__ import annotations
from typing import TYPE_CHECKING, Any
from .acquisition import ThompsonAcqOptimizer, UCBAcqOptimizer

if TYPE_CHECKING:
    from .protocols import AcquisitionOptimizer, Surrogate


def build_surrogate(cfg: Any) -> Surrogate:
    from .enn_surrogate import ENNSurrogate
    from .gp_surrogate import GPSurrogate
    from .no_surrogate import NoSurrogate
    from ..config.optimizer_config import OptimizerConfig

    if isinstance(cfg, OptimizerConfig):
        cfg = cfg.surrogate

    name = type(cfg).__name__
    if name == "ENNSurrogateConfig":
        return ENNSurrogate(cfg)
    if name == "GPSurrogateConfig":
        return GPSurrogate()
    if name == "NoSurrogateConfig":
        return NoSurrogate()
    raise ValueError(f"Unknown surrogate config type: {name}")


def build_acquisition_optimizer(cfg: Any) -> AcquisitionOptimizer:
    from .acquisition import ParetoAcqOptimizer, RandomAcqOptimizer
    from ..config.optimizer_config import OptimizerConfig

    if isinstance(cfg, OptimizerConfig):
        cfg = cfg.acquisition

    name = type(cfg).__name__
    if name == "DrawAcquisitionConfig":
        return ThompsonAcqOptimizer()
    if name == "ParetoAcquisitionConfig":
        return ParetoAcqOptimizer()
    if name == "RandomAcquisitionConfig":
        return RandomAcqOptimizer()
    if name == "UCBAcquisitionConfig":
        return UCBAcqOptimizer()
    raise ValueError(f"Unknown acquisition config type: {name}")


def build_trust_region(
    cfg: Any, num_dim: int, rng: Any, candidate_rv: Any = None
) -> Any:
    from .incumbent_selector import ScalarIncumbentSelector
    from ..morbo_trust_region import MorboTrustRegion
    from ..no_trust_region import NoTrustRegion
    from ..turbo_trust_region import TurboTrustRegion
    from ..config.optimizer_config import OptimizerConfig

    if isinstance(cfg, OptimizerConfig):
        candidate_rv = cfg.candidate_rv
        cfg = cfg.trust_region

    name = type(cfg).__name__
    if name == "MorboTRConfig":
        return MorboTrustRegion(
            config=cfg, num_dim=num_dim, rng=rng, candidate_rv=candidate_rv
        )
    if name == "TurboTRConfig":
        return TurboTrustRegion(
            config=cfg,
            num_dim=num_dim,
            incumbent_selector=ScalarIncumbentSelector(noise_aware=cfg.noise_aware),
        )
    if name == "NoTRConfig":
        return NoTrustRegion(
            config=cfg,
            num_dim=num_dim,
            incumbent_selector=ScalarIncumbentSelector(noise_aware=cfg.noise_aware),
        )
    raise ValueError(f"Unknown trust region config type: {name}")
