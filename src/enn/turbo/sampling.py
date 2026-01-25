from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from . import turbo_utils

if TYPE_CHECKING:
    from numpy.random import Generator


def draw_lhd(*, bounds: np.ndarray, num_arms: int, rng: Generator) -> np.ndarray:
    bounds = np.asarray(bounds, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError(f"bounds must be (d, 2), got {bounds.shape}")
    num_arms = int(num_arms)
    if num_arms <= 0:
        raise ValueError(num_arms)
    num_dim = int(bounds.shape[0])
    return turbo_utils.from_unit(
        turbo_utils.latin_hypercube(num_arms, num_dim, rng=rng),
        bounds,
    )
