import numpy as np

from enn._rust import ackley_core as _ackley_core


def ackley_core(
    x: np.ndarray, a: float = 20.0, b: float = 0.2, c: float = 2 * np.pi
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    return np.asarray(_ackley_core(x, float(a), float(b), float(c)), dtype=float)
