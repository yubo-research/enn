"""Metric sources for the streaming metric-learning models of ``evals/metric_12d.py``.

Each source sees every added row (``observe``) and returns metric weights at a checkpoint
(``weights``); ``StreamedModel`` applies them with ``MBPANNMetric.set_weights``.

- ``bpann_disk_ml_reservoir``: the LOOCV L-BFGS-B fit of ``bpann_disk_metric_learning``, but on a
  1,000-row reservoir sample of the whole stream instead of a fresh random subsample.
- ``bpann_disk_sobol`` / ``bpann_disk_corr``: ``validated_dependence_weights`` of the reservoir; no optimization.
- ``bpann_disk_spsa_row``: ``PerturbationMetricLearner`` called once per added row (a step per row), from
  ``SCALE_X`` distances; ``bpann_disk_spsa_sobol_row`` starts it from the Sobol weights at 100 rows.
- ``bpann_disk_spsa`` / ``bpann_disk_spsa_sobol``: the same learner called once per added batch, so a
  batch's per-row steps are computed from the state at the start of the batch and summed (cheaper).

The per-row models cost ~0.6 ms per row (~10 min per 1e6 rows), so they are not in the evals' default model list.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np

from enn.enn.metric_stream import (
    PerturbationMetricLearner,
    Reservoir,
    validated_dependence_weights,
)

SPSA_INIT_ROWS = 100


class ReservoirLbfgs:
    """L-BFGS-B LOOCV fit on a reservoir; warm-started once the reservoir was full at the previous fit."""

    def __init__(
        self,
        core: ModuleType,
        num_dim: int,
        capacity: int,
        k: int,
        rng: np.random.Generator,
    ) -> None:
        self._core, self._k = core, k
        self.reservoir = Reservoir(capacity, num_dim, rng)
        self._theta: np.ndarray | None = None
        self._was_full = False

    def observe(self, x: np.ndarray, y: np.ndarray) -> None:
        self.reservoir.add(x, y)

    def weights(self) -> np.ndarray:
        rx, ry = self.reservoir.x, self.reservoir.y
        metric = self._core.Metric(rx.shape[1])
        k = min(self._k, len(ry) - 1)
        if self._was_full and self._theta is not None:
            metric.theta = self._theta.copy()
            self._core.fit_exact(metric, rx, ry, k, outer=1, restart=False)
        else:
            self._core.fit_exact(metric, rx, ry, k)
        self._theta = metric.theta.copy()
        self._was_full = len(ry) == self.reservoir.capacity
        return metric.a


class ReservoirDependence:
    def __init__(
        self, index: str, num_dim: int, capacity: int, k: int, rng: np.random.Generator
    ) -> None:
        self._index, self._k = index, k
        self.reservoir = Reservoir(capacity, num_dim, rng)

    def observe(self, x: np.ndarray, y: np.ndarray) -> None:
        self.reservoir.add(x, y)

    def weights(self) -> np.ndarray:
        return validated_dependence_weights(
            self.reservoir.x, self.reservoir.y, self._k, self._index
        )


class Spsa:
    def __init__(
        self,
        num_dim: int,
        capacity: int,
        k: int,
        rng: np.random.Generator,
        init_rows: int,
        per_row: bool,
    ) -> None:
        self.learner = PerturbationMetricLearner(
            num_dim, rng, k=k, capacity=capacity, init_rows=init_rows
        )
        self._per_row = per_row

    def observe(self, x: np.ndarray, y: np.ndarray) -> None:
        if not self._per_row:
            self.learner.update(x, y)
            return
        for i in range(len(y)):
            self.learner.update(x[i : i + 1], y[i : i + 1])

    def weights(self) -> np.ndarray:
        return self.learner.weights


FAST_STREAM_METRIC_MODELS: tuple[str, ...] = (
    "bpann_disk_ml_reservoir",
    "bpann_disk_sobol",
    "bpann_disk_corr",
    "bpann_disk_spsa",
    "bpann_disk_spsa_sobol",
)
PER_ROW_SPSA_MODELS: tuple[str, ...] = (
    "bpann_disk_spsa_row",
    "bpann_disk_spsa_sobol_row",
)
STREAM_METRIC_MODELS = FAST_STREAM_METRIC_MODELS + PER_ROW_SPSA_MODELS
SPSA_MODELS = STREAM_METRIC_MODELS[3:]


def make_source(
    name: str,
    core: ModuleType,
    num_dim: int,
    capacity: int,
    k: int,
    rng: np.random.Generator,
) -> ReservoirLbfgs | ReservoirDependence | Spsa:
    if name == "bpann_disk_ml_reservoir":
        return ReservoirLbfgs(core, num_dim, capacity, k, rng)
    if name in ("bpann_disk_sobol", "bpann_disk_corr"):
        return ReservoirDependence(
            "sobol" if name == "bpann_disk_sobol" else "correlation",
            num_dim,
            capacity,
            k,
            rng,
        )
    if name in SPSA_MODELS:
        init_rows = SPSA_INIT_ROWS if "sobol" in name else 0
        return Spsa(num_dim, capacity, k, rng, init_rows, per_row=name.endswith("_row"))
    raise ValueError(f"unknown stream metric model {name!r}")
