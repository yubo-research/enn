import itertools
import json
import sys
import tempfile
from dataclasses import replace

import numpy as np

from evals.harness import metric_12d, metric_ranges
from evals.harness.metric_12d import Metric12dConfig, run_model


def wrap(data, mode):
    def f(n, rng):
        x, y = data(n, rng)
        if mode == "f32":
            x = x.astype(np.float32).astype(np.float64)
        return x, y

    return f


def shifted(data):
    def f(n, rng):
        x, y = data(n, rng)
        return 1000.0 + 1e-2 * x, y

    return f


def run_one(s, pn, d, mode, cfg, fh):
    c = replace(cfg, seed=s)

    def data(n, rng):
        x, y = d(n, rng)
        if mode == "f32" and n == c.stream_rows():
            x = x.astype(np.float32).astype(np.float64)
        return x, y

    with tempfile.TemporaryDirectory() as wd:
        res = run_model("bpann_disk_auto", wd, c, data)
    for r in res:
        fh.write(
            json.dumps(
                {
                    "seed": s,
                    "prob": pn,
                    "mode": mode,
                    "n": r.num_obs,
                    "loglik": r.loglik,
                    "nrmse": r.nrmse,
                }
            )
            + "\n"
        )
        fh.flush()


def main():
    seeds = int(sys.argv[1])
    out = sys.argv[2]
    cfg = Metric12dConfig(
        n_grid=(100, 1000, 10000, 30000),
        num_test=500,
        num_seeds=1,
        num_fit_candidates=50,
        num_fit_samples=50,
    )
    probs = {
        "sparse": metric_12d.make_data,
        "ranges": metric_ranges.make_data,
        "shifted": shifted(metric_12d.make_data),
    }
    with open(out, "a") as fh:
        for s, (pn, d), mode in itertools.product(
            range(seeds), probs.items(), ("f64", "f32")
        ):
            run_one(s, pn, d, mode, cfg, fh)


main()
