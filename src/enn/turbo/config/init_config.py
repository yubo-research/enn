from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .init_strategy_base import InitStrategy


@dataclass(frozen=True)
class InitConfig:
    init_strategy: InitStrategy | None = None
    num_init: int | None = None

    def __post_init__(self) -> None:
        if self.init_strategy is not None and not isinstance(
            self.init_strategy, InitStrategy
        ):
            raise ValueError(
                f"init_strategy must be an InitStrategy, got {self.init_strategy!r}"
            )
        if self.num_init is not None and self.num_init <= 0:
            raise ValueError(f"num_init must be > 0, got {self.num_init}")

    def get_init_strategy(self) -> Any:
        if self.init_strategy is not None:
            return self.init_strategy
        from .init_strategies.hybrid_init import HybridInit

        return HybridInit()
