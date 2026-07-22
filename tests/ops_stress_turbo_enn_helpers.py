"""Shared fakes for turbo-enn stress tests (keeps concrete types out of the test module)."""

from __future__ import annotations

import numpy as np


class RecordingTurboOpt:
    """Minimal optimizer stub that records ask/tell sizes (and optional event tags)."""

    def __init__(
        self,
        *,
        tell_sizes: list[int] | None = None,
        events: list[str] | None = None,
        ask_returns_used: list[object] | None = None,
        num_dim: int = 2,
    ) -> None:
        self.tell_sizes = tell_sizes if tell_sizes is not None else []
        self.events = events
        self.ask_returns_used = ask_returns_used
        self.num_dim = num_dim

    def ask(self, num_arms: int = 1):
        if self.events is not None:
            self.events.append(f"ask:{num_arms}")
        if self.ask_returns_used is not None:
            arms = np.full((1, self.num_dim), 99.0, dtype=float)
            self.ask_returns_used.append(arms)
            return arms
        return np.zeros((1, self.num_dim), dtype=float)

    def tell(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        n = int(x.shape[0])
        if self.events is not None:
            self.events.append(f"tell:{n}")
        self.tell_sizes.append(n)
        assert y.shape == (n, 1)
        assert x.shape[1] == self.num_dim


class RecordingObjective:
    """Zero-valued objective that optionally records eval batch sizes."""

    bounds = [-1.0, 1.0]

    def __init__(self, *, events: list[str] | None = None) -> None:
        self.events = events

    def __call__(self, x):
        x = np.asarray(x)
        if self.events is not None:
            self.events.append(f"eval:{x.shape[0]}")
        return np.zeros(x.shape[0], dtype=float)


def unit_bounds(num_dim: int = 2) -> np.ndarray:
    return np.array([[-1.0, 1.0]] * num_dim, dtype=float)


def collect_tell_sizes(
    *,
    num_obs: int,
    num_ask: int,
    tell_all: bool,
    seed_chunk: int | None = None,
) -> list[int]:
    """Run ``run_turbo_enn_stress`` with recording fakes; return tell batch sizes."""
    from enn.turbo.config.enn_index_driver import ENNIndexDriver
    from ops.stress import run_turbo_enn_stress

    opt = RecordingTurboOpt()
    kwargs: dict = {
        "index_driver": ENNIndexDriver.FLAT,
        "num_dim": 2,
        "num_obs": num_obs,
        "num_ask": num_ask,
        "optimizer": opt,
        "objective": RecordingObjective(),
        "bounds": unit_bounds(2),
        "tell_all": tell_all,
    }
    if seed_chunk is not None:
        kwargs["seed_chunk"] = seed_chunk
    list(run_turbo_enn_stress(**kwargs))
    return opt.tell_sizes
