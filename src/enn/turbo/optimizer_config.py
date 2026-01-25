from __future__ import annotations
from . import config as _config

__all__ = list(_config.__all__)


def __getattr__(name: str) -> object:
    return getattr(_config, name)
