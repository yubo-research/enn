"""Stream FLAT, FLAT+scale_x, BPANN_DISK+NONE and BPANN_DISK+AUTO through one eval.

Usage: PYTHONPATH=src:. python reports/metric_learning_2/run.py {12d|ranges} FIRST_SEED NUM_SEEDS > OUT
Seeds and data are those of ``./ops/evaluate.py run short/metric_12d`` / ``short/metric_ranges``.
Several processes with disjoint seeds can run side by side; timings are then contended.
"""

import sys

from evals import metric_12d, metric_ranges

MODELS = ("flat", "flat_scale_x", "bpann_disk", "bpann_disk_auto")


def main() -> None:
    tag, first, num = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    cfg = metric_12d.Metric12dConfig(seed=first, num_seeds=num)
    if tag == "12d":
        metric_12d.run_eval(cfg, models=MODELS)
    elif tag == "ranges":
        metric_ranges.run_eval(cfg, models=MODELS)
    else:
        raise SystemExit(f"unknown eval {tag!r}; use 12d or ranges")


if __name__ == "__main__":
    main()
