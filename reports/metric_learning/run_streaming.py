"""Stream the reservoir / dependence / SPSA metric models, plus BPANN_DISK NONE and METRIC_LEARNING as controls.

Usage: PYTHONPATH=src:. python reports/metric_learning/run_streaming.py {12d|ranges} FIRST_SEED NUM_SEEDS [MODEL ...] > OUT
Without MODEL arguments it runs the controls and the batched streaming models (``MODELS``).
Seeds and data are those of ``./ops/evaluate.py run short/metric_12d`` / ``short/metric_ranges``, so the
controls can be checked against the earlier runs. Several processes with disjoint seeds run side by side.
"""

import sys

from evals import metric_12d, metric_ranges
from evals.metric_stream_models import FAST_STREAM_METRIC_MODELS

MODELS = ("bpann_disk", "bpann_disk_metric_learning", *FAST_STREAM_METRIC_MODELS)


def main() -> None:
    tag, first, num = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    models = tuple(sys.argv[4:]) or MODELS
    cfg = metric_12d.Metric12dConfig(seed=first, num_seeds=num)
    if tag == "12d":
        metric_12d.run_eval(cfg, models=models)
    elif tag == "ranges":
        metric_ranges.run_eval(cfg, models=models)
    else:
        raise SystemExit(f"unknown eval {tag!r}; use 12d or ranges")


if __name__ == "__main__":
    main()
