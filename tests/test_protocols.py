from typing import Protocol
from enn.turbo.components.acquisition_optimizer_protocol import AcquisitionOptimizer
from enn.turbo.components.incumbent_selector_protocol import IncumbentSelector
from enn.turbo.components.surrogate_protocol import Surrogate
from enn.turbo.components.trust_region_protocol import TrustRegion
from enn.enn.enn_like_protocol import ENNLike


def test_protocols_are_protocols():
    assert issubclass(AcquisitionOptimizer, Protocol)
    assert issubclass(IncumbentSelector, Protocol)
    assert issubclass(Surrogate, Protocol)
    assert issubclass(TrustRegion, Protocol)
    assert issubclass(ENNLike, Protocol)
