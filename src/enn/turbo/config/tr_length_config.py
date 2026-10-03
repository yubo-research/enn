from __future__ import annotations

from dataclasses import dataclass


def _reject_nonpositive(name: str, value: float | None) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def _reject_order(
    length_init: float | None,
    length_min: float | None,
    length_max: float | None,
) -> None:
    if length_min is not None and length_max is not None and length_min >= length_max:
        raise ValueError(
            f"length_min must be < length_max, got {length_min} >= {length_max}"
        )
    if length_init is not None and length_max is not None and length_init > length_max:
        raise ValueError(
            f"length_init must be <= length_max, got {length_init} > {length_max}"
        )
    if length_min is not None and length_init is not None and length_min > length_init:
        raise ValueError(
            f"length_min must be <= length_init, got {length_min} > {length_init}"
        )


@dataclass(frozen=True)
class TRLengthConfig:
    """Lengths the caller set. ``None`` means Rust's default."""

    length_init: float | None = None
    length_min: float | None = None
    length_max: float | None = None

    def __post_init__(self) -> None:
        _reject_nonpositive("length_init", self.length_init)
        _reject_nonpositive("length_min", self.length_min)
        _reject_nonpositive("length_max", self.length_max)
        _reject_order(self.length_init, self.length_min, self.length_max)
