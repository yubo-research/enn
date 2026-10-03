"""Replay the BPANN_DISK scale_x update rule on the eval streams; count rescales and rebuilds.

Mirrors ``apply_incremental_x_scale`` (rust/crates/ennbo/src/model/metric.rs): after each add,
the new scale is the population std of all rows so far (1 for n < 2); nothing happens if every
scale is within ``RESCALE_TOL`` (log) of the applied one, else the index is rescaled in place, or
re-partitioned if some scale is more than ``REBUILD_DRIFT`` (log) from the partition's.
Usage: PYTHONPATH=src:. python reports/metric_learning/scale_x_updates.py
"""

import numpy as np

from evals import metric_12d, metric_ranges

RESCALE_TOL = 0.01
REBUILD_DRIFT = float(np.log(2.0))
SEEDS = range(10)


def batch_bounds(config: metric_12d.Metric12dConfig) -> list[int]:
    """Row count after each add, as in ``StreamedModel.advance``."""
    stops, lo = [], 0
    for hi in config.n_grid:
        stops += [min(hi, s + config.batch) for s in range(lo, hi, config.batch)]
        lo = hi
    return stops


def scales_at(x: np.ndarray, stops: list[int]) -> np.ndarray:
    """Population std of ``x[:n]`` for each ``n`` in ``stops`` (1 where n < 2 or std ~ 0)."""
    zero = np.zeros((1, x.shape[1]))
    s1 = np.vstack([zero, np.cumsum(x, axis=0)])[stops]
    s2 = np.vstack([zero, np.cumsum(x * x, axis=0)])[stops]
    n = np.asarray(stops, dtype=float)[:, None]
    std = np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2, 0.0))
    return np.where((n >= 2) & (std > 1e-12), std, 1.0)


def replay(x: np.ndarray, stops: list[int]) -> tuple[list[int], list[int]]:
    """Row counts at which the index was rescaled, and at which it was re-partitioned."""
    scales = scales_at(x, stops)
    applied = built = scales[0]
    rescales, rebuilds = [], []
    for n, new in zip(stops[1:], scales[1:]):
        if np.max(np.abs(np.log(new / applied))) <= RESCALE_TOL:
            continue
        if np.max(np.abs(np.log(new / built))) > REBUILD_DRIFT:
            rebuilds.append(n)
            built = new
        else:
            rescales.append(n)
        applied = new
    return rescales, rebuilds


def main() -> None:
    config = metric_12d.Metric12dConfig()
    stops = batch_bounds(config)
    print(f"adds per stream: {len(stops) - 1} (after the model is built at n={stops[0]})")
    for tag, data in (("12d", metric_12d.make_data), ("ranges", metric_ranges.make_data)):
        for seed in SEEDS:
            x, _ = data(max(config.n_grid), np.random.default_rng(seed))
            rescales, rebuilds = replay(x, stops)
            print(
                f"[{tag}] seed={seed} rescales={len(rescales)} last_rescale_n={max(rescales, default=0)} "
                f"rebuilds={len(rebuilds)} rebuild_n={rebuilds}"
            )


if __name__ == "__main__":
    main()
