from __future__ import annotations

from dataclasses import dataclass

from .candidate_rv import CandidateRV
from .raasp_driver import RAASPDriver


def _reject_negative(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")


@dataclass(frozen=True)
class CandidateGenConfig:
    """Candidate-pool parameters. Rust computes the pool size."""

    candidate_rv: CandidateRV = CandidateRV.SOBOL
    min_candidates: int = 10
    max_candidates: int = 5000
    num_candidates_per_dim: int = 100
    num_candidates_per_arm: int = 0
    raasp_driver: RAASPDriver = RAASPDriver.ORIG

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_rv, CandidateRV):
            raise ValueError(
                f"candidate_rv must be a CandidateRV enum, got {self.candidate_rv!r}"
            )
        _reject_negative("min_candidates", self.min_candidates)
        _reject_negative("max_candidates", self.max_candidates)
        _reject_negative("num_candidates_per_dim", self.num_candidates_per_dim)
        _reject_negative("num_candidates_per_arm", self.num_candidates_per_arm)
