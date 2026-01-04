from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NoSurrogateConfig:
    pass


@dataclass(frozen=True)
class GPSurrogateConfig:
    pass


@dataclass(frozen=True)
class ENNSurrogateConfig:
    k: int | None = None
    num_fit_samples: int | None = None
    num_fit_candidates: int | None = None
    scale_x: bool = False

    def __post_init__(self) -> None:
        if self.num_fit_samples is not None and self.num_fit_samples <= 0:
            raise ValueError(f"num_fit_samples must be > 0, got {self.num_fit_samples}")
        if self.num_fit_candidates is not None and self.num_fit_candidates <= 0:
            raise ValueError(
                f"num_fit_candidates must be > 0, got {self.num_fit_candidates}"
            )


SurrogateConfig = NoSurrogateConfig | GPSurrogateConfig | ENNSurrogateConfig
