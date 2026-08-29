from __future__ import annotations

from typing import Any


class InitStrategy:
    def create_runtime_strategy(
        self,
        *,
        bounds: Any,
        rng: Any,
        num_init: int | None,
    ) -> Any:
        raise NotImplementedError("InitStrategy.create_runtime_strategy")
