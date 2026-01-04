from __future__ import annotations

import importlib

from .enn.enn import EpistemicNearestNeighbors
from .enn.enn_fit import enn_fit

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "TurboOptimizer": (".turbo.turbo_optimizer", "TurboOptimizer"),
    "Telemetry": (".turbo.turbo_utils", "Telemetry"),
    "Turbo": (".turbo.turbo_optimizer", "TurboOptimizer"),
    "TurboOneConfig": (".turbo.turbo_config", "TurboOneConfig"),
    "TurboZeroConfig": (".turbo.turbo_config", "TurboZeroConfig"),
    "TurboENNConfig": (".turbo.turbo_config", "TurboENNConfig"),
    "LHDOnlyConfig": (".turbo.turbo_config", "LHDOnlyConfig"),
}


def __getattr__(name: str):
    spec = _LAZY_ATTRS.get(name)
    if spec is not None:
        module_name, attr_name = spec
        module = importlib.import_module(module_name, __package__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__: list[str] = [
    "EpistemicNearestNeighbors",
    "enn_fit",
    *_LAZY_ATTRS.keys(),
]
