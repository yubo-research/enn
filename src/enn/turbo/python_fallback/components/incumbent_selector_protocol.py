from __future__ import annotations

from typing import Any, Callable, Protocol


class IncumbentSelector(Protocol):
    select: Callable[[Any, Any | None, Any], int]
    reset: Callable[[Any], None]
