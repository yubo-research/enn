from __future__ import annotations

import importlib
from typing import Any


def lazy_getattr(
    *,
    name: str,
    module_name: str,
    package: str,
    mapping: dict[str, tuple[str, str]],
    extra: str,
) -> Any:
    spec = mapping.get(name)
    if spec is None:
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
    rel_module, attr_name = spec
    try:
        module = importlib.import_module(rel_module, package)
        return getattr(module, attr_name)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(f"{e}. Install extras via {extra}.") from e
