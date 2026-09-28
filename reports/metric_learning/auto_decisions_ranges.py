"""Replay the bpann_disk_auto stream of short/metric_ranges for each seed and print AUTO's decision.

Uses reports/metric_12d/auto_decisions.py with the different-ranges data.
Usage: PYTHONPATH=src:. python reports/metric_learning/auto_decisions_ranges.py > auto_decisions_ranges.out
"""

import importlib.util
import os

from evals.metric_ranges import make_data

HERE = os.path.dirname(os.path.abspath(__file__))
REPLAY_PATH = os.path.join(HERE, "..", "metric_12d", "auto_decisions.py")


def main():
    spec = importlib.util.spec_from_file_location("auto_decisions", REPLAY_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main(make_data)


if __name__ == "__main__":
    main()
