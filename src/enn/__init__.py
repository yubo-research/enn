from __future__ import annotations

from .enn import EpistemicNearestNeighbors, enn_fit

_LAZY_IMPORTS = ("TurboMode", "TurboOptimizer", "Turbo", "Telemetry")


def _lazy_load(name: str):
    from . import turbo

    return getattr(turbo, name)


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        return _lazy_load(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__: list[str] = [
    "EpistemicNearestNeighbors",
    "enn_fit",
    *_LAZY_IMPORTS,
]
