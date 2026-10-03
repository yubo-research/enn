"""How far are BPANN_DISK's returned neighbors, compared with exact and random rows?

Mislabeled row ids would put returned neighbors at random-row distance; an approximate
but correctly labeled search puts them just beyond the exact k-th neighbor. The last
column is recall against the exact 10-NN within the batch the first returned neighbor
came from.

Usage: PYTHONPATH=src python reports/mbpann/bpann_recall_dist.py [BATCH]
"""

import sys
import tempfile

import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams, PosteriorFlags
from enn.turbo.config.enn_index_driver import ENNIndexDriver

N, D, K, Q = 40000, 10, 10, 500


def returned_neighbors(x: np.ndarray, y: np.ndarray, q: np.ndarray, batch: int) -> np.ndarray:
    m = EpistemicNearestNeighbors(
        x[:batch], y[:batch], index_driver=ENNIndexDriver.BPANN_DISK, work_dir=tempfile.mkdtemp()
    )
    m.ensure_index_sync()
    for lo in range(batch, N, batch):
        m.add(x[lo : lo + batch], y[lo : lo + batch])
        m.ensure_index_sync()
    p = ENNParams(k_num_neighbors=K, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.1)
    return np.asarray(m.batch_posterior(q, [p], flags=PosteriorFlags()).idx)


def query_stats(x: np.ndarray, v: np.ndarray, a: np.ndarray, batch: int) -> tuple[float, ...]:
    d2 = ((x - v) ** 2).sum(1)
    e = np.argpartition(d2, K)[:K]
    lo = (int(a[0]) // batch) * batch
    own = set((np.argpartition(d2[lo : lo + batch], K)[:K] + lo).tolist())
    got = set(a.tolist())
    return (
        len(got & set(e.tolist())) / K,
        float(np.sqrt(d2[a]).mean()),
        float(np.sqrt(d2[e]).mean()),
        float(np.sqrt(d2).mean()),
        len(own & got) / K,
    )


def main(batch: int) -> None:
    rng = np.random.default_rng(0)
    x, y, q = rng.random((N, D)), rng.random((N, 1)), rng.random((Q, D))
    idx = returned_neighbors(x, y, q, batch)
    rec, ret, exa, rnd, own = np.mean([query_stats(x, v, a, batch) for v, a in zip(q, idx)], axis=0)
    print(
        f"batch={batch} recall={rec:.3f} mean_dist returned={ret:.3f} "
        f"exact={exa:.3f} random={rnd:.3f} recall_vs_exact_within_first_returned_batch={own:.3f}"
    )


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2000)
