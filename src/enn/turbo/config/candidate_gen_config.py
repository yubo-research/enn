from __future__ import annotations

from dataclasses import dataclass

from .candidate_rv import CandidateRV
from .raasp_driver import RAASPDriver


@dataclass(frozen=True)
class CandidateGenConfig:
    candidate_rv: CandidateRV = CandidateRV.SOBOL
    num_candidates: int | None = None
    num_candidates_per_arm: int | None = None
    raasp_driver: RAASPDriver = RAASPDriver.ORIG

    def resolve_num_candidates(self, *, num_dim: int, num_arms: int) -> int:
        base = (
            self.num_candidates
            if self.num_candidates is not None
            else min(5000, 100 * int(num_dim))
        )
        if self.num_candidates_per_arm is not None:
            base = max(base, self.num_candidates_per_arm * int(num_arms))
        return int(base)

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_rv, CandidateRV):
            raise ValueError(
                f"candidate_rv must be a CandidateRV enum, got {self.candidate_rv!r}"
            )
        if self.num_candidates is not None and self.num_candidates <= 0:
            raise ValueError(f"num_candidates must be > 0, got {self.num_candidates}")
        if self.num_candidates_per_arm is not None and self.num_candidates_per_arm <= 0:
            raise ValueError(
                f"num_candidates_per_arm must be > 0, got {self.num_candidates_per_arm}"
            )
