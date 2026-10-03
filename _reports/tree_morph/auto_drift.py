"""Print every AUTO metric change (drift, re-partition or not, relative scales) on the seed-0 stream of one eval.

Batching is that of bench.py / evals/metric_12d.py. Usage:
PYTHONPATH=src:. python reports/tree_morph/auto_drift.py {12d|ranges} > reports/tree_morph/auto_drift_TAG.out
"""

import sys
import tempfile

import numpy as np

from enn.enn import mbpann
from evals import metric_12d, metric_ranges
from reports.bpann_disk_writeup.bench import stream_rows

SET_WEIGHTS = mbpann.MBPANNMetric.set_weights


def logged_set_weights(self, weights):
    w = np.asarray(weights, dtype=float)
    drift = self.drift(w)
    rebuild = SET_WEIGHTS(self, weights)
    scales = np.array2string(np.sqrt(w / w.max()), precision=3, max_line_width=200)
    print(
        f"rows={self.reservoir.num_seen} drift={drift:.3f} (ln2={np.log(2):.3f}) rebuild={rebuild} rel_scales={scales}",
        flush=True,
    )
    return rebuild


def main() -> None:
    mbpann.MBPANNMetric.set_weights = logged_set_weights
    make_data = {"12d": metric_12d.make_data, "ranges": metric_ranges.make_data}[sys.argv[1]]
    cfg = metric_12d.Metric12dConfig()
    x, y = make_data(max(cfg.n_grid), np.random.default_rng(0))
    with tempfile.TemporaryDirectory() as work_dir:
        model, lo = None, 0
        for hi in cfg.n_grid:
            model = stream_rows(model, "bpann_disk_auto", work_dir, x, y, lo=lo, hi=hi, batch=cfg.batch)
            print(f"checkpoint n={hi} rebuilds={model.metric.num_rebuilds}", flush=True)
            lo = hi


if __name__ == "__main__":
    main()
