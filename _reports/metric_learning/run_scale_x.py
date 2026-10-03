"""Stream only BPANN_DISK NONE and SCALE_X through one eval; the other models come from earlier runs.

Usage: PYTHONPATH=src:. python reports/metric_learning/run_scale_x.py {12d|ranges} > OUT
``bpann_disk`` is rerun so its loglik can be checked against the earlier run (same seeds, same data)
and so its times are measured next to ``bpann_disk_scale_x``'s.
"""

import sys

from evals import metric_12d, metric_ranges

MODELS = ("bpann_disk", "bpann_disk_scale_x")


def main() -> None:
    tag = sys.argv[1]
    if tag == "12d":
        metric_12d.run_eval(models=MODELS)
    elif tag == "ranges":
        metric_ranges.run_eval(models=MODELS)
    else:
        raise SystemExit(f"unknown eval {tag!r}; use 12d or ranges")


if __name__ == "__main__":
    main()
