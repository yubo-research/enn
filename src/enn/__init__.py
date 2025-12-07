from .enn import EpistemicNearestNeighbors
from .enn_fit import enn_fit


def __getattr__(name: str):
    """Lazy import for backwards compatibility with turbo classes."""
    if name in ("TurboMode", "TurboOptimizer", "Turbo", "Telemetry"):
        import turbo

        if name == "TurboMode":
            return turbo.TurboMode
        elif name == "TurboOptimizer":
            return turbo.TurboOptimizer
        elif name == "Turbo":
            return turbo.Turbo
        elif name == "Telemetry":
            return turbo.Telemetry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__: list[str] = [
    "EpistemicNearestNeighbors",
    "TurboMode",
    "TurboOptimizer",
    "Turbo",
    "Telemetry",
    "enn_fit",
]
