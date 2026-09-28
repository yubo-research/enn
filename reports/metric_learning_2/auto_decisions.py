"""Replay the bpann_disk_auto stream of run.py and print AUTO's state at each checkpoint.

For each checkpoint: number of refits, the latest leave-one-out gain, whether the learned metric is
applied, and the applied weights relative to their largest (12d: the first two inputs matter;
ranges: the ideal weights are proportional to 1 / range^2, printed as ``ideal``).
Usage: PYTHONPATH=src:. python reports/metric_learning_2/auto_decisions.py 0,1,2 > auto_decisions.out
"""

import sys
import tempfile

import numpy as np

from evals import metric_12d, metric_ranges

DATA = {"12d": metric_12d.make_data, "ranges": metric_ranges.make_data}


def replay(tag, seed):
    cfg = metric_12d.Metric12dConfig(seed=seed)
    x, y = DATA[tag](max(cfg.n_grid), np.random.default_rng(seed))
    with tempfile.TemporaryDirectory() as work_dir:
        streamed = metric_12d.StreamedModel("bpann_disk_auto", work_dir, cfg)
        lo = 0
        for hi in cfg.n_grid:
            for start in range(lo, hi, cfg.batch):
                streamed._add_rows(x[start : min(hi, start + cfg.batch)], y[start : min(hi, start + cfg.batch)])
            m = streamed.model.metric
            rel = m.weights / m.weights.max()
            gain = "na" if m.heldout_gain is None else f"{m.heldout_gain:+.3f}"
            print(
                f"{tag} seed={seed} n={hi} refits={m.num_refits} gain={gain} on={int(m.uses_learned_metric)} "
                f"rebuilds={m.num_rebuilds} rescales={m.num_rescales} rel_w={np.array2string(rel, precision=3)}",
                flush=True,
            )
            lo = hi


def main():
    ideal = metric_ranges.RANGES[0] ** 2 / metric_ranges.RANGES**2
    print(f"ranges ideal rel_w={np.array2string(ideal, precision=3)}")
    for seed in (int(s) for s in sys.argv[1].split(",")):
        for tag in DATA:
            replay(tag, seed)


if __name__ == "__main__":
    main()
