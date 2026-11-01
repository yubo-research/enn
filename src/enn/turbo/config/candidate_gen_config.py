from __future__ import annotations

from dataclasses import dataclass

from enn._rust import validate_candidate_counts

from .candidate_rv import CandidateRV
from .raasp_driver import RAASPDriver


@dataclass(frozen=True)
class CandidateGenConfig:
    candidate_rv: CandidateRV | None = None
    min_candidates: int = 10
    max_candidates: int = 5000
    num_candidates_per_dim: int = 100
    num_candidates_per_arm: int = 0
    raasp_driver: RAASPDriver = RAASPDriver.ORIG

    def __post_init__(self) -> None:
        if self.candidate_rv is not None and not isinstance(
            self.candidate_rv, CandidateRV
        ):
            raise ValueError(
                f"candidate_rv must be a CandidateRV enum, got {self.candidate_rv!r}"
            )
        validate_candidate_counts(
            self.min_candidates,
            self.max_candidates,
            self.num_candidates_per_dim,
            self.num_candidates_per_arm,
        )
