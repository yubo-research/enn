import numpy as np
from numpy.random import Generator

from .ackley_class import numpy_normal
from .ackley_core import ackley_core


class DoubleAckley:
    def __init__(self, noise: float, rng: Generator):
        self.noise = noise
        self.rng = rng
        self._normal = numpy_normal(rng)
        self.bounds = [-32.768, 32.768]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        n, d = x.shape
        if d % 2 != 0:
            raise ValueError("num_dim must be even for DoubleAckley")
        mid = d // 2
        y1 = -ackley_core(x[:, :mid])
        y2 = -ackley_core(x[:, mid:])
        if self.noise != 0.0:
            eps1 = np.asarray(self._normal.standard_normals(n), dtype=float)
            eps2 = np.asarray(self._normal.standard_normals(n), dtype=float)
            y1 = y1 + self.noise * eps1
            y2 = y2 + self.noise * eps2
        return np.stack([y1, y2], axis=1)
