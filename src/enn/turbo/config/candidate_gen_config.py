from __future__ import annotations
from dataclasses import dataclass, field
from .candidate_rv import CandidateRV
from .raasp_driver import RAASPDriver
from .num_candidates_fn import NumCandidatesFn, default_num_candidates


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
