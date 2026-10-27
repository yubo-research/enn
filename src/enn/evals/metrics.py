"""Shared predictive scores and directed metric formatting."""

import numpy as np

from ops.qa import gaussian_nll


def gaussian_loglik(y: np.ndarray, mu: np.ndarray, se: np.ndarray) -> float:
    """Mean Gaussian predictive log-likelihood (larger is better)."""
    return float(-gaussian_nll(y, mu, se))


def rmse(y: np.ndarray, mu: np.ndarray) -> float:
    err = np.asarray(mu, dtype=float).ravel() - np.asarray(y, dtype=float).ravel()
    return float(np.sqrt(np.mean(err**2)))


def format_plain(name: str, value: object) -> str:
    return f"{name} = {value}"


def format_larger(name: str, value: object) -> str:
    return f"LARGER({name}) = {value}"


def format_smaller(name: str, value: object) -> str:
    return f"SMALLER({name}) = {value}"


def format_directed(direction: str, name: str, value: object) -> str:
    if direction == "LARGER":
        return format_larger(name, value)
    if direction == "SMALLER":
        return format_smaller(name, value)
    return format_plain(name, value)
