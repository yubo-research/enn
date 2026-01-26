from typing import Protocol
from enn.turbo.components import AcquisitionOptimizer
from enn.turbo.components import IncumbentSelector
from enn.turbo.components import Surrogate
from enn.turbo.components import TrustRegion
from enn.enn.enn_like_protocol import ENNLike


def test_protocols_are_protocols():
    assert issubclass(AcquisitionOptimizer, Protocol)
    assert issubclass(IncumbentSelector, Protocol)
    assert issubclass(Surrogate, Protocol)
    assert issubclass(TrustRegion, Protocol)
    assert issubclass(ENNLike, Protocol)
