from typing import Protocol

from enn.turbo.components import (
    AcquisitionOptimizer,
    IncumbentSelector,
    Surrogate,
    TrustRegion,
)


def test_protocols_are_protocols():
    assert issubclass(AcquisitionOptimizer, Protocol)
    assert issubclass(IncumbentSelector, Protocol)
    assert issubclass(Surrogate, Protocol)
    assert issubclass(TrustRegion, Protocol)
