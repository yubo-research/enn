import numpy as np

from enn._rust import DoubleAckley as _RustDoubleAckley


class DoubleAckley:
    def __init__(self, noise: float, seed: int):
        self._inner = _RustDoubleAckley(float(noise), seed)

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
