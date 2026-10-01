from .draw_acquisition_config import DrawAcquisitionConfig
from .nds_optimizer_config import NDSOptimizerConfig
from .pareto_acquisition_config import ParetoAcquisitionConfig
from .raasp_optimizer_config import RAASPOptimizerConfig
from .random_acquisition_config import RandomAcquisitionConfig
from .ucb_acquisition_config import UCBAcquisitionConfig

AcquisitionConfig = (
    UCBAcquisitionConfig
    | DrawAcquisitionConfig
    | ParetoAcquisitionConfig
    | RandomAcquisitionConfig
)


def acquisition_kind(acq: object) -> str:
    """Name passed to the Rust optimizer. Unknown classes are an error."""
    if isinstance(acq, UCBAcquisitionConfig):
        return "ucb"
    if isinstance(acq, DrawAcquisitionConfig):
        return "thompson"
    if isinstance(acq, ParetoAcquisitionConfig):
        return "pareto"
    if isinstance(acq, RandomAcquisitionConfig):
        return "random"
    raise TypeError(
        f"acquisition must be an AcquisitionConfig, got {type(acq).__name__}"
    )


AcqOptimizerConfig = RAASPOptimizerConfig | NDSOptimizerConfig
__all__ = [
    "AcqOptimizerConfig",
    "AcquisitionConfig",
    "acquisition_kind",
    "DrawAcquisitionConfig",
    "NDSOptimizerConfig",
    "ParetoAcquisitionConfig",
    "RAASPOptimizerConfig",
    "RandomAcquisitionConfig",
    "UCBAcquisitionConfig",
]
