from __future__ import annotations

from dataclasses import dataclass

from .candidate_rv import CandidateRV


@dataclass(frozen=True)
class CandidateGenConfig:
    candidate_rv: CandidateRV = CandidateRV.SOBOL
    num_candidates: int = 5000

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_rv, CandidateRV):
            raise ValueError(
                f"candidate_rv must be a CandidateRV enum, got {self.candidate_rv!r}"
            )
        if self.num_candidates <= 0:
            raise ValueError(f"num_candidates must be > 0, got {self.num_candidates}")
