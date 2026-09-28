"""Replay the bpann_disk_auto stream of short/metric_12d for each seed and print AUTO's decision.

Same data and fit random streams as the eval, so decisions match the eval's AUTO runs.
Test-set queries are skipped. Usage: python reports/metric_12d/auto_decisions.py > auto_decisions.out
"""

import tempfile
from dataclasses import replace

import numpy as np

import evals.metric_12d as mod


def replay(cfg, data=mod.make_data):
    rng = np.random.default_rng(cfg.seed)
    x, y = data(max(cfg.n_grid), rng)
    with tempfile.TemporaryDirectory(prefix=mod.WORK_DIR_PREFIX) as work_dir:
        streamed = mod.StreamedModel("bpann_disk_auto", work_dir, cfg)
        lo = 0
        for hi in cfg.n_grid:
            streamed.advance(x, y, lo, hi)
            fit = streamed.metric_fit
            print(
                f"seed={cfg.seed} n={hi} gain={fit.heldout_gain:.4f} accepted={fit.theta is not None}",
                flush=True,
            )
            lo = hi


def main(data=mod.make_data):
    base = mod.Metric12dConfig()
    for i in range(base.num_seeds):
        replay(replace(base, seed=base.seed + i), data)


if __name__ == "__main__":
    main()
