"""Where fit time goes, and how repeatable query time is (BPANN_DISK+NONE, irrelevant-inputs data, seed 0).

For N in (1e5, 1e6): stream the rows as bench.py does, then time
- ``train_rows_at(range(N))``, the full-table copy that batch-mode ``enn_fit`` makes first,
- ``enn_fit`` itself,
- five posterior calls on the same 1,000 test points (ms per point).

Usage: PYTHONPATH=src:. python reports/bpann_disk_writeup/check_fit_query.py > check_fit_query.out
"""

import tempfile
import time

import numpy as np

from enn.enn.enn_fit import enn_fit
from evals import metric_12d
from ops.stress import DRAW_FLAGS


def main() -> None:
    rng = np.random.default_rng(0)
    x, y = metric_12d.make_data(1_000_000, rng)
    x_test, _ = metric_12d.make_data(1000, rng)
    for n in (100_000, 1_000_000):
        with tempfile.TemporaryDirectory(prefix="enn_bpann_check_") as work_dir:
            model = metric_12d.build_model("bpann_disk", x[:500], y[:500], work_dir)
            for start in range(500, n, 500):
                model.add(x[start : start + 500], y[start : start + 500])
                model.ensure_index_sync()
            t0 = time.perf_counter()
            model.train_rows_at(list(range(len(model))))
            t_copy = time.perf_counter() - t0
            t0 = time.perf_counter()
            params = enn_fit(model, k=10, num_fit_candidates=100, num_fit_samples=100, rng=np.random.default_rng(1))
            t_fit = time.perf_counter() - t0
            qs = []
            for _ in range(5):
                t0 = time.perf_counter()
                model.posterior(x_test, params=params, flags=DRAW_FLAGS)
                qs.append(1e3 * (time.perf_counter() - t0) / len(x_test))
            print(
                f"n = {n} copy_s = {t_copy:.3f} fit_s = {t_fit:.3f} "
                f"query_ms = {' '.join(f'{q:.4f}' for q in qs)} median = {np.median(qs):.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
