"""Cost of one metric change, and of queries afterwards, versus n.

Each index is streamed in batches (ensure_index_sync after each), then the metric
changes once. BPANN_DISK has to build a new model on sqrt(w) * x; BPANN_DISK + metric learning
rescales in place or re-indexes in place.

Usage: PYTHONPATH=src python reports/mbpann/timing_mbpann.py OUT.json
"""

import json
import sys
import tempfile
import time

import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams, PosteriorFlags
from enn.turbo.config.enn_index_driver import ENNIndexDriver
from enn.turbo.config.enn_x_scaling import ENNMetricLearning

D = 10
K = 10
N_QUERY = 1000
SIZES = (40_000, 160_000, 640_000)
BATCHES = (500, 2000)


def stream(x, y, batch, metric_learning):
    model = EpistemicNearestNeighbors(
        x[:batch], y[:batch], metric_learning=metric_learning, index_driver=ENNIndexDriver.BPANN_DISK, work_dir=tempfile.mkdtemp(prefix="mbt_")
    )
    for lo in range(batch, len(x), batch):
        model.add(x[lo : lo + batch], y[lo : lo + batch])
        model.ensure_index_sync()
    model.ensure_index_sync()
    return model


def timed_query(model, xq):
    p = ENNParams(k_num_neighbors=K, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.1)
    t0 = time.perf_counter()
    idx = model.batch_posterior(xq, [p], flags=PosteriorFlags()).idx
    return time.perf_counter() - t0, np.asarray(idx)


def recall(idx, x, xq, w):
    hits = 0
    for q, row in zip(xq, idx):
        exact = np.argpartition(((x - q) ** 2) @ w, K)[:K]
        hits += len(set(row.tolist()) & set(exact.tolist()))
    return hits / (K * len(xq))


def one(n, batch, rng):
    x, y = rng.random((n, D)), rng.random((n, 1))
    xq = rng.random((N_QUERY, D))
    w = np.exp(rng.uniform(-2, 2, D))
    rows = []
    for how in ("rescale", "rebuild_in_place", "bpann_copy"):
        metric_learning = ENNMetricLearning.OFF if how == "bpann_copy" else ENNMetricLearning.ON
        model = stream(x, y, batch, metric_learning)
        t0 = time.perf_counter()
        if how == "bpann_copy":
            model = stream(x * np.sqrt(w), y, n, metric_learning)
            q_scale = np.sqrt(w)
        else:
            model.rust_backend.set_metric_scale(1.0 / np.sqrt(w), rebuild=how != "rescale")
            q_scale = np.ones(D)
        change = time.perf_counter() - t0
        q_sec, idx = timed_query(model, xq * q_scale)
        rows.append(dict(n=n, batch=batch, how=how, change_sec=change, query_sec=q_sec, recall=recall(idx, x, xq, w)))
        print(rows[-1], flush=True)
    return rows


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    out = []
    for batch in BATCHES:
        for n in SIZES:
            out += one(n, batch, rng)
            with open(sys.argv[1], "w") as fh:
                json.dump(out, fh)
