from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:

    class NumCandidatesFn:
        def __call__(self, *, num_dim: int, num_arms: int) -> int: ...
else:
    NumCandidatesFn = Any


def default_num_candidates(*, num_dim: int, num_arms: int) -> int:
    return min(5000, 100 * int(num_dim))


def const_num_candidates(n: int) -> NumCandidatesFn:
    n = int(n)
    if n <= 0:
        raise ValueError(f"num_candidates must be > 0, got {n}")

    def fn(*, num_dim: int, num_arms: int) -> int:
        return n

    return fn
