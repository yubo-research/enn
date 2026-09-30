"""Build data files and the table for metric_12d.tex from seeds10.out.

seeds10.out is the output of `./ops/evaluate.py run short/metric_12d` (10 seeds). Every value
is a mean ± standard error over seeds, computed from the per-seed lines.
Usage: python reports/metric_12d/make_metric_12d.py
"""

import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_FILE = os.path.join(HERE, "seeds10.out")
SEED_LINE = re.compile(
    r"^seed = (\d+) model = (\w+) n = (\d+) loglik = (\S+) nrmse = (\S+) add_s = (\S+) query_s = (\S+)$",
    re.MULTILINE,
)
MODELS = [
    ("flat", "FLAT, \\texttt{NONE}"),
    ("flat_scale_x", "FLAT, \\texttt{SCALE\\_X}"),
    ("bpann_disk", "BPANN\\_DISK, \\texttt{NONE}"),
    ("bpann_disk_metric_learning", "BPANN\\_DISK, \\texttt{METRIC\\_LEARNING}"),
    ("bpann_disk_auto", "BPANN\\_DISK, \\texttt{AUTO}"),
]
METRICS = ("loglik", "nrmse", "add", "q")


def mean_se(values):
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size))


def load_seeds():
    """{(model, n): array of shape (num_seeds, 4)} with columns loglik, nrmse, add_s, query_s."""
    with open(RUN_FILE) as f:
        text = f.read()
    rows = {}
    for m in SEED_LINE.finditer(text):
        seed, model, n, *vals = m.groups()
        rows.setdefault((model, int(n)), {})[int(seed)] = [float(v) for v in vals]
    seeds = sorted(next(iter(rows.values())))
    if any(sorted(r) != seeds for r in rows.values()):
        raise ValueError(
            "every (model, n) must have the same seeds; is the run complete?"
        )
    return {k: np.array([r[s] for s in seeds]) for k, r in rows.items()}, len(seeds)


def ns_of(data, model):
    return sorted(n for m, n in data if m == model)


def write_data_files(data):
    header = " ".join(f"{c} {c}_se" for c in METRICS)
    for model, _ in MODELS:
        with open(os.path.join(HERE, f"d_{model}.dat"), "w") as f:
            f.write(f"n {header}\n")
            for n in ns_of(data, model):
                cols = [mean_se(data[(model, n)][:, i]) for i in range(len(METRICS))]
                f.write(
                    f"{n} " + " ".join(f"{mu:.5g} {se:.5g}" for mu, se in cols) + "\n"
                )


def pm(stat, fmt):
    return f"${stat[0]:{fmt}} \\pm {stat[1]:{fmt.lstrip('+')}}$"


def write_table(data):
    lines = []
    for model, label in MODELS:
        ns = ns_of(data, model)
        last = data[(model, ns[-1])]
        total_add = sum(data[(model, n)][:, 2] for n in ns)
        lines.append(
            f"{label} & {pm(mean_se(last[:, 0]), '+.2f')} & {pm(mean_se(last[:, 1]), '.2f')} & "
            f"{pm(mean_se(total_add), '.1f')} & {pm(mean_se(last[:, 3]), '.2f')}\\\\"
        )
    with open(os.path.join(HERE, "t_final.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def print_summary(data):
    """Per-checkpoint numbers quoted in the text, including paired (same-seed) differences."""
    pairs = [
        ("bpann_disk_metric_learning", "bpann_disk"),
        ("bpann_disk_auto", "bpann_disk"),
        ("bpann_disk_auto", "bpann_disk_metric_learning"),
        ("flat", "bpann_disk"),
        ("flat_scale_x", "flat"),
    ]
    for n in ns_of(data, "flat"):
        parts = [f"{m}={mean_se(data[(m, n)][:, 0])[0]:+.3f}" for m, _ in MODELS]
        diffs = [
            "{}-{}={:+.3f}±{:.3f}".format(
                a, b, *mean_se(data[(a, n)][:, 0] - data[(b, n)][:, 0])
            )
            for a, b in pairs
        ]
        adds = [f"add_med_{m}={np.median(data[(m, n)][:, 2]):.3f}" for m, _ in MODELS]
        qs = [f"q_med_{m}={np.median(data[(m, n)][:, 3]):.3f}" for m, _ in MODELS]
        print(f"n={n} " + " ".join(parts + diffs + adds + qs))
    for model, _ in MODELS:
        ns = ns_of(data, model)
        total_add = sum(data[(model, n)][:, 2] for n in ns)
        print(
            f"{model}: median total add {np.median(total_add):.1f} s, "
            f"median query at n={ns[-1]} {np.median(data[(model, ns[-1])][:, 3]):.2f} s, "
            f"loglik per seed at n={ns[-1]} {np.round(data[(model, ns[-1])][:, 0], 2).tolist()}"
        )


def main():
    data, num_seeds = load_seeds()
    print(f"seeds={num_seeds}")
    write_data_files(data)
    write_table(data)
    print_summary(data)


if __name__ == "__main__":
    main()
