import numpy as np
from numpy.random import Generator
from .ackley_core import ackley_core


class DoubleAckley:
    def __init__(self, noise: float, rng: Generator):
        self.noise = noise
        self.rng = rng
        self.bounds = [-32.768, 32.768]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        n, d = x.shape
        if d % 2 != 0:
            raise ValueError("num_dim must be even for DoubleAckley")
        mid = d // 2
        x1 = x[:, :mid]
        x2 = x[:, mid:]
        y1 = -ackley_core(x1) + self.noise * self.rng.normal(size=n)
        y2 = -ackley_core(x2) + self.noise * self.rng.normal(size=n)
        return np.stack([y1, y2], axis=1)
