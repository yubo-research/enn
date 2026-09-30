"""Diagnose BPANN_DISK's low recall: which part of the search loses the true neighbors.

Same setup as bpann_recall_repro.py (40k uniform rows, d=10, batches of 2000, synced
after every batch). Each variant sets BPANN tuning through a temporary config file.
For each variant we report recall@10 and the mean number of distinct 2000-row batches
the returned neighbors come from (the exact neighbors come from about 9.9 batches).

Usage: PYTHONPATH=src python reports/mbpann/bpann_recall_diag.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

from enn._rust import set_config_path
from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams, PosteriorFlags
from enn.turbo.config.enn_index_driver import ENNIndexDriver

N, D, B, K, Q = 40000, 10, 2000, 10, 500

CONFIG = """[bpann]
search_fragment_budget_max = {budget}
search_beam_width = {beam}
"""


def build(x: np.ndarray, y: np.ndarray, sync_first: bool) -> EpistemicNearestNeighbors:
    m = EpistemicNearestNeighbors(
        x[:B], y[:B], index_driver=ENNIndexDriver.BPANN_DISK, work_dir=tempfile.mkdtemp()
    )
    if sync_first:
        m.ensure_index_sync()
    for lo in range(B, N, B):
        m.add(x[lo : lo + B], y[lo : lo + B])
        m.ensure_index_sync()
    return m


def run(budget: int, beam: int, order: str, sync_first: bool = True) -> tuple[float, float]:
    cfg = Path(tempfile.mkdtemp()) / "config.toml"
    cfg.write_text(CONFIG.format(budget=budget, beam=beam))
    set_config_path(str(cfg))
    rng = np.random.default_rng(0)
    x, y, q = rng.random((N, D)), rng.random((N, 1)), rng.random((Q, D))
    if order == "sorted":
        x = x[np.argsort(x[:, 0])]
    m = build(x, y, sync_first)
    p = ENNParams(k_num_neighbors=K, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.1)
    idx = np.asarray(m.batch_posterior(q, [p], flags=PosteriorFlags()).idx)
    exact = [np.argpartition(((x - v) ** 2).sum(1), K)[:K] for v in q]
    rec = float(np.mean([len(set(a) & set(b)) / K for a, b in zip(idx, exact)]))
    nbatch = float(np.mean([len(set(a // B)) for a in idx]))
    set_config_path(None)
    return rec, nbatch


VARIANTS = [
    (1, 1, "random", True),
    (64, 1, "random", True),
    (1, 16, "random", True),
    (64, 16, "random", True),
    (64, 64, "random", True),
    (1, 1, "random", False),
    (1, 1, "sorted", True),
]

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x, q = rng.random((N, D)), rng.random((Q, D))
    ex = [np.argpartition(((x - v) ** 2).sum(1), K)[:K] for v in q]
    print(f"exact neighbors: distinct batches {np.mean([len(set(e // B)) for e in ex]):.2f}")
    for budget, beam, order, sync_first in VARIANTS:
        rec, nb = run(budget, beam, order, sync_first)
        print(
            f"budget={budget:3d} beam={beam:3d} order={order:6s} sync_first={sync_first!s:5s} "
            f"recall={rec:.3f} distinct_batches={nb:.2f}",
            flush=True,
        )
    sys.exit(0)
