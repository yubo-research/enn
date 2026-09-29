"""Stream FLAT, FLAT+scale_x, BPANN_DISK+NONE and BPANN_DISK+AUTO through one metric eval, timing add, fit and query apart.

Usage: PYTHONPATH=src:. python reports/tree_morph/bench.py {12d|ranges} FIRST_SEED NUM_SEEDS > OUT

Data, seeds, batches (500 rows, ``ensure_index_sync`` after each) and hyperparameter fits are those
of ``evals/metric_12d.py`` / ``evals/metric_ranges.py``. Per checkpoint it prints one line with
- ``add_s``: wall time of every ``add`` + ``ensure_index_sync`` since the previous checkpoint
  (for AUTO this includes its reservoir updates, metric refits and in-place rescales),
- ``fit_s``: wall time of the ``enn_fit`` hyperparameter fit at this checkpoint,
- ``query_s``: wall time of one ``posterior`` call on the 1,000 test points.
"""

import sys
import tempfile
import time

import numpy as np

from enn.enn.enn_fit import enn_fit
from evals import metric_12d, metric_ranges
from evals.flat_sphere import gaussian_loglik, rmse
from ops.stress import DRAW_FLAGS

MODELS = ("flat", "flat_scale_x", "bpann_disk", "bpann_disk_auto")
DATA = {"12d": metric_12d.make_data, "ranges": metric_ranges.make_data}


def stream_rows(model, name, work_dir, x, y, *, lo, hi, batch):
    """Add rows ``lo:hi`` in batches with a sync after each; build the model on the first batch."""
    for start in range(lo, hi, batch):
        stop = min(hi, start + batch)
        if model is None:
            model = metric_12d.build_model(name, x[start:stop], y[start:stop], work_dir)
        else:
            model.add(x[start:stop], y[start:stop])
        model.ensure_index_sync()
    return model


def run_model(name, work_dir, cfg, make_data, seed):
    rng = np.random.default_rng(seed)
    x, y = make_data(max(cfg.n_grid), rng)
    x_test, y_test = make_data(cfg.num_test, rng)
    y_std = float(np.std(y_test))
    fit_rng = np.random.default_rng(seed + 1)
    model, lo = None, 0
    for hi in cfg.n_grid:
        t0 = time.perf_counter()
        model = stream_rows(model, name, work_dir, x, y, lo=lo, hi=hi, batch=cfg.batch)
        t1 = time.perf_counter()
        params = enn_fit(
            model, k=cfg.k, num_fit_candidates=cfg.num_fit_candidates, num_fit_samples=cfg.num_fit_samples, rng=fit_rng
        )
        t2 = time.perf_counter()
        post = model.posterior(x_test, params=params, flags=DRAW_FLAGS)
        t3 = time.perf_counter()
        metric = model.metric
        extra = (
            ""
            if metric is None
            else f" refits = {metric.num_refits} rescales = {metric.num_rescales} rebuilds = {metric.num_rebuilds}"
            f" learned = {int(metric.uses_learned_metric)}"
        )
        print(
            f"seed = {seed} model = {name} n = {hi} num_added = {hi - lo} "
            f"loglik = {gaussian_loglik(y_test, post.mu, post.se):.4f} nrmse = {rmse(y_test, post.mu) / y_std:.4f} "
            f"add_s = {t1 - t0:.5f} fit_s = {t2 - t1:.5f} query_s = {t3 - t2:.5f}{extra}",
            flush=True,
        )
        lo = hi


def main() -> None:
    tag, first, num = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    cfg = metric_12d.Metric12dConfig()
    print(f"eval={tag} seeds={first}..{first + num - 1} models={','.join(MODELS)} batch={cfg.batch} k={cfg.k}", flush=True)
    for seed in range(first, first + num):
        with tempfile.TemporaryDirectory(prefix="enn_tree_morph_") as work_dir:
            for name in MODELS:
                run_model(name, work_dir, cfg, DATA[tag], seed)


if __name__ == "__main__":
    main()
