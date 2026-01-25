from .draw_acquisition_config import DrawAcquisitionConfig
from .hnr_optimizer_config import HnROptimizerConfig
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
AcqOptimizerConfig = RAASPOptimizerConfig | HnROptimizerConfig | NDSOptimizerConfig
__all__ = [
    "AcqOptimizerConfig",
    "AcquisitionConfig",
    "DrawAcquisitionConfig",
    "HnROptimizerConfig",
    "NDSOptimizerConfig",
    "ParetoAcquisitionConfig",
    "RAASPOptimizerConfig",
    "RandomAcquisitionConfig",
    "UCBAcquisitionConfig",
]
