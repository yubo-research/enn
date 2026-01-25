from __future__ import annotations
from dataclasses import dataclass
from .init_strategies import HybridInit, InitStrategy


@dataclass(frozen=True)
class InitConfig:
    init_strategy: InitStrategy = HybridInit()
    num_init: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.init_strategy, InitStrategy):
            raise ValueError(
                f"init_strategy must be an InitStrategy, got {self.init_strategy!r}"
            )
        if self.num_init is not None and self.num_init <= 0:
            raise ValueError(f"num_init must be > 0, got {self.num_init}")
