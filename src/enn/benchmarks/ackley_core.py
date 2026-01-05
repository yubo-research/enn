import numpy as np


def ackley_core(
    x: np.ndarray, a: float = 20.0, b: float = 0.2, c: float = 2 * np.pi
) -> np.ndarray:
    if x.ndim == 1:
        x = x[None, :]
    x = x - 1
    term1 = -a * np.exp(-b * np.sqrt((x**2).mean(axis=1)))
    term2 = -np.exp(np.cos(c * x).mean(axis=1))
    return term1 + term2 + a + np.e
