"""Reproduce BPANN_DISK's build-history-dependent recall (no metric change involved).

Same 40k rows (d=10), batches of 2000. Syncing right after construction makes every
fragment a k-means tree and recall@10 is about 0.01; skipping that first sync
leaves a flat-forest fragment, which forces a search of all fragments, and recall is about 0.3.

Usage: PYTHONPATH=src python reports/mbpann/bpann_recall_repro.py
"""

import tempfile

import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams, PosteriorFlags
from enn.turbo.config.enn_index_driver import ENNIndexDriver

N, D, B, K = 40000, 10, 2000, 10


def recall(sync_first: bool) -> float:
    rng = np.random.default_rng(0)
    x, y, q = rng.random((N, D)), rng.random((N, 1)), rng.random((500, D))
    m = EpistemicNearestNeighbors(
        x[:B], y[:B], index_driver=ENNIndexDriver.BPANN_DISK, work_dir=tempfile.mkdtemp()
    )
    if sync_first:
        m.ensure_index_sync()
    for lo in range(B, N, B):
        m.add(x[lo : lo + B], y[lo : lo + B])
        m.ensure_index_sync()
    p = ENNParams(k_num_neighbors=K, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.1)
    idx = np.asarray(m.batch_posterior(q, [p], flags=PosteriorFlags()).idx)
    exact = [np.argpartition(((x - v) ** 2).sum(1), K)[:K] for v in q]
    return float(np.mean([len(set(a) & set(b)) / K for a, b in zip(idx, exact)]))


if __name__ == "__main__":
    print("sync after construction:", recall(True))
    print("no sync after construction:", recall(False))
