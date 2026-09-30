"""Recall@10 of BPANN_DISK and BPANN_DISK + metric learning when rows are indexed by background flushes.

15,000 rows (d=6) arrive in batches of 500, with schedule_background_flush after every
third batch. MBPANN runs are given 2 random metric changes per batch (rescale only, or
re-index on every change). Recall is measured against exact neighbors under each model's
final metric.

Usage: PYTHONPATH=src python reports/mbpann/flush_recall_repro.py
"""

import tempfile

import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams, PosteriorFlags
from enn.enn.mbpann import MBPANNMetric
from enn.turbo.config.enn_index_driver import ENNIndexDriver
from enn.turbo.config.enn_x_scaling import ENNMetricLearning

N, D, B, K = 15000, 6, 500, 10


def recall(metric_learning, rebuild_drift=None, flush=True):
    rng = np.random.default_rng(3)
    x, y, q = rng.random((N, D)), rng.random((N, 1)), rng.random((300, D))
    m = EpistemicNearestNeighbors(x[:B], y[:B], metric_learning=metric_learning, index_driver=ENNIndexDriver.BPANN_DISK, work_dir=tempfile.mkdtemp())
    metric = None if rebuild_drift is None else MBPANNMetric(m, rebuild_drift=rebuild_drift)
    for i, lo in enumerate(range(B, N, B)):
        m.add(x[lo : lo + B], y[lo : lo + B])
        if flush and i % 3 == 0:
            m.schedule_background_flush()
        if metric is not None:
            for _ in range(2):
                metric.set_weights(np.exp(rng.uniform(-3, 3, D)))
    w = np.ones(D) if metric is None else metric.weights
    p = ENNParams(k_num_neighbors=K, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.1)
    idx = np.asarray(m.batch_posterior(q, [p], flags=PosteriorFlags()).idx)
    exact = [np.argsort(((x - v) ** 2) @ w)[:K] for v in q]
    return float(np.mean([len(set(a) & set(b)) / K for a, b in zip(idx, exact)]))


if __name__ == "__main__":
    print("BPANN_DISK, no metric, background flush:", recall(ENNMetricLearning.OFF))
    print("BPANN_DISK + metric learning, rescale, background flush:", recall(ENNMetricLearning.ON, np.inf))
    print("BPANN_DISK + metric learning, rescale, no flush:", recall(ENNMetricLearning.ON, np.inf, flush=False))
    print("BPANN_DISK + metric learning, re-index every change, background flush:", recall(ENNMetricLearning.ON, 0.0))
