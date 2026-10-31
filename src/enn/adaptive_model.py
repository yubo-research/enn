"""Public description of the candidate selected by native adaptation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AdaptChoice:
    geometry: str
    weight: float
    output: str
    neighbors: int
    alpha: float
