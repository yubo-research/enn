"""Stream FLAT, FLAT+scale_x, BPANN_DISK+NONE and BPANN_DISK+AUTO through one metric eval, timing add, fit and query apart.

Usage: PYTHONPATH=src:. python reports/tree_morph/bench.py {12d|ranges} FIRST_SEED NUM_SEEDS > OUT

The protocol and output lines are those of ``reports/bpann_disk_writeup/bench.py`` (whose ``run_model`` this
reuses), with FLAT+scale_x added: data, seeds, batches (500 rows, ``ensure_index_sync`` after each) and
hyperparameter fits of ``evals/metric_12d.py`` / ``evals/metric_ranges.py``; per checkpoint ``add_s``
(every ``add`` + sync since the previous checkpoint, including AUTO's refits, rescales and morph),
``fit_s`` (``enn_fit``) and ``query_s`` (one ``posterior`` call on the 1,000 test points).
"""

import sys
import tempfile

from evals import metric_12d
from reports.bpann_disk_writeup.bench import DATA, run_model

MODELS = ("flat", "flat_scale_x", "bpann_disk", "bpann_disk_auto")


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
