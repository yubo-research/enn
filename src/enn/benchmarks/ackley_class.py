import numpy as np
from numpy.random import Generator

from enn._rust import Ackley as _RustAckley


class Ackley:
    """Facade over the Rust Ackley benchmark. The only field is the Rust object."""

    def __init__(self, noise: float, rng: Generator):
        seed = int(rng.integers(0, 2**63 - 1))
        self._inner = _RustAckley(float(noise), seed)

    @property
    def noise(self) -> float:
        return float(self._inner.noise)

    @property
    def bounds(self) -> list[float]:
        return [float(v) for v in self._inner.bounds]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        return np.asarray(self._inner.evaluate(x), dtype=float)
