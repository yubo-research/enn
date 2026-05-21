from typing import Protocol

from enn.turbo.python_fallback.components.incumbent_selector_protocol import (
    IncumbentSelector,
)
from enn.turbo.python_fallback.components.protocols import (
    AcquisitionOptimizer,
    Surrogate,
    TrustRegion,
)


def test_protocols_are_protocols():
    assert issubclass(AcquisitionOptimizer, Protocol)
    assert issubclass(IncumbentSelector, Protocol)
    assert issubclass(Surrogate, Protocol)
    assert issubclass(TrustRegion, Protocol)
