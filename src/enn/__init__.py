from __future__ import annotations
import importlib
from .enn.enn_class import EpistemicNearestNeighbors
from .enn.enn_fit import enn_fit

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "create_optimizer": (".turbo.optimizer", "create_optimizer"),
    "Telemetry": (".turbo.turbo_utils", "Telemetry"),
    "OptimizerConfig": (".turbo.optimizer_config", "OptimizerConfig"),
    "turbo_one_config": (".turbo.optimizer_config", "turbo_one_config"),
    "turbo_zero_config": (".turbo.optimizer_config", "turbo_zero_config"),
    "turbo_enn_config": (".turbo.optimizer_config", "turbo_enn_config"),
    "lhd_only_config": (".turbo.optimizer_config", "lhd_only_config"),
    "TurboTRConfig": (".turbo.config.trust_region", "TurboTRConfig"),
    "MorboTRConfig": (".turbo.config.trust_region", "MorboTRConfig"),
    "NoTRConfig": (".turbo.config.trust_region", "NoTRConfig"),
    "CandidateRV": (".turbo.optimizer_config", "CandidateRV"),
    "InitStrategy": (".turbo.optimizer_config", "InitStrategy"),
    "AcqType": (".turbo.optimizer_config", "AcqType"),
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
