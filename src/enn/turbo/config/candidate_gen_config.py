from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from .candidate_rv import CandidateRV
from .raasp_driver import RAASPDriver

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


@dataclass(frozen=True)
class CandidateGenConfig:
    candidate_rv: CandidateRV = CandidateRV.SOBOL
    num_candidates: NumCandidatesFn = field(
        default_factory=lambda: default_num_candidates
    )
    raasp_driver: RAASPDriver = RAASPDriver.ORIG

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_rv, CandidateRV):
            raise ValueError(
                f"candidate_rv must be a CandidateRV enum, got {self.candidate_rv!r}"
            )
        if not callable(self.num_candidates):
            raise ValueError(
                f"num_candidates must be callable, got {type(self.num_candidates)!r}"
            )
        test_n = int(self.num_candidates(num_dim=1, num_arms=1))
        if test_n <= 0:
            raise ValueError(f"num_candidates must be > 0, got {test_n}")
