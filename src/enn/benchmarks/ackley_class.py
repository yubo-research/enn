import numpy as np
from numpy.random import Generator
from .ackley_core import ackley_core


class Ackley:
    def __init__(self, noise: float, rng: Generator):
        self.noise = noise
        self.rng = rng
        self.bounds = [-32.768, 32.768]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        y = -ackley_core(x) + self.noise * self.rng.normal(size=(x.shape[0],))
        return y if y.ndim > 0 else float(y)
