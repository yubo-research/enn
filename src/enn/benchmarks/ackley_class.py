import numpy as np
from numpy.random import Generator

from enn._rust import NumpyNormal

from .ackley_core import ackley_core


def numpy_normal(rng: Generator) -> NumpyNormal:
    state = rng.bit_generator.state["state"]
    bits = int(state["state"])
    inc = int(state["inc"])
    mask = (1 << 64) - 1
    return NumpyNormal(bits >> 64, bits & mask, inc >> 64, inc & mask)


class Ackley:
    def __init__(self, noise: float, rng: Generator):
        self.noise = noise
        self.rng = rng
        self._normal = numpy_normal(rng)
        self.bounds = [-32.768, 32.768]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        base = -ackley_core(x)
        if self.noise == 0.0:
            return base
        eps = np.asarray(self._normal.standard_normals(x.shape[0]), dtype=float)
        return base + self.noise * eps
