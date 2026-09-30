"""Build data files and the table for report-metric_learning-2.tex.

- runs/{12d,ranges}_s*.out: ``run.py TAG SEED 2`` for seeds 0, 2, 4, 6, 8 (10 seeds, run side by side)
- timing_{12d,ranges}.out: ``run.py TAG 0 1``, run alone (fit and query times)

Accuracy values are means ± standard errors over the 10 seeds. Times come from the uncontended
seed-0 run: fit time is the sum of ``add_s`` over all checkpoints (adds, syncs, AUTO's metric
refits, hyperparameter fits), query time is ``query_s`` at the last checkpoint (1,000 test points).
The seed-0 timing run must reproduce seed 0 of the 10-seed run.
Usage: python reports/metric_learning_2/make_report.py
"""

import glob
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SEED_LINE = re.compile(
    r"^seed = (\d+) model = (\w+) n = (\d+) loglik = (\S+) nrmse = (\S+) add_s = (\S+) query_s = (\S+)$",
    re.MULTILINE,
)
MODELS = [
    ("flat", "FLAT"),
    ("flat_scale_x", "FLAT + \\code{scale\\_x}"),
    ("bpann_disk", "BPANN\\_DISK + \\code{NONE}"),
    ("bpann_disk_auto", "BPANN\\_DISK + \\code{AUTO}"),
]
TAGS = ("12d", "ranges")
METRICS = ("loglik", "nrmse", "add", "q")
PAIRS = [("bpann_disk_auto", m) for m, _ in MODELS[:3]]


def mean_se(values):
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else float("nan")


def load_seeds(paths):
    """{(model, n): array of shape (num_seeds, 4)} with columns loglik, nrmse, add_s, query_s."""
    text = "".join(open(p).read() for p in paths)
    rows = {}
    for m in SEED_LINE.finditer(text):
        seed, model, n, *vals = m.groups()
        rows.setdefault((model, int(n)), {})[int(seed)] = [float(v) for v in vals]
    seeds = sorted(next(iter(rows.values())))
    if any(sorted(r) != seeds for r in rows.values()) or len(rows) != 11 * len(MODELS):
        raise ValueError(f"{paths}: every (model, n) must have the same seeds; is the run complete?")
    return {k: np.array([r[s] for s in seeds]) for k, r in rows.items()}, seeds


def ns_of(data, model):
    return sorted(n for m, n in data if m == model)


def write_data_files(tag, data):
    """d_TAG_MODEL.dat: one row per checkpoint, columns n and (mean, se) of each metric, for pgfplots."""
    header = "n " + " ".join(f"{c} {c}_se" for c in METRICS)
    for model, _ in MODELS:
        table = [[n, *np.ravel([mean_se(col) for col in data[(model, n)].T])] for n in ns_of(data, model)]
        np.savetxt(os.path.join(HERE, f"d_{tag}_{model}.dat"), table, fmt=["%d"] + ["%.5g"] * 2 * len(METRICS), header=header, comments="")


def cost(timing, model):
    ns = ns_of(timing, model)
    return float(sum(timing[(model, n)][0, 2] for n in ns)), float(timing[(model, ns[-1])][0, 3])


def pm(stat, fmt):
    return f"${stat[0]:{fmt}} \\pm {stat[1]:{fmt.lstrip('+')}}$"


def write_table(data_by_tag, timing_by_tag):
    """One row per model: loglik and nrmse at 1e6 (10 seeds), fit and query time (seed 0), for each problem."""
    lines = []
    for model, label in MODELS:
        cells = []
        for tag in TAGS:
            d = data_by_tag[tag][(model, 1000000)]
            fit, query = cost(timing_by_tag[tag], model)
            cells += [pm(mean_se(d[:, 0]), "+.2f"), pm(mean_se(d[:, 1]), ".3f"), f"${fit:.0f}$", f"${query:.2f}$"]
        lines.append(f"{label} & " + " & ".join(cells) + "\\\\")
    with open(os.path.join(HERE, "t_main.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def print_summary(tag, data, timing):
    """Numbers quoted in the text: per-checkpoint means, paired (same-seed) AUTO differences, costs."""
    for n in ns_of(data, "flat"):
        parts = ["{}={:+.3f}±{:.3f}/{:.3f}".format(m, *mean_se(data[(m, n)][:, 0]), mean_se(data[(m, n)][:, 1])[0]) for m, _ in MODELS]
        diffs = [
            "{}-{}={:+.3f}±{:.3f}#{}".format(
                a, b, *mean_se(data[(a, n)][:, 0] - data[(b, n)][:, 0]), int(np.sum(data[(a, n)][:, 0] > data[(b, n)][:, 0]))
            )
            for a, b in PAIRS
        ]
        print(f"[{tag}] n={n} " + " ".join(parts + diffs))
    for model, _ in MODELS:
        fit, query = cost(timing, model)
        adds = " ".join(f"{timing[(model, n)][0, 2]:.2f}" for n in ns_of(timing, model))
        print(f"[{tag}] seed-0 {model}: fit total {fit:.2f} s, query at 1e6 {query:.3f} s; add_s per checkpoint {adds}")


def check_timing_run(tag, data, seeds, timing):
    """Seed 0 of the timing run must reproduce seed 0 of the 10-seed run (loglik and nrmse)."""
    i = seeds.index(0)
    diff = max(float(np.abs(timing[k][0, :2] - data[k][i, :2]).max()) for k in timing)
    print(f"[{tag}] timing run vs 10-seed run, seed 0: max |diff| of loglik and nrmse = {diff:.2g}")


def main():
    data_by_tag, timing_by_tag = {}, {}
    for tag in TAGS:
        data, seeds = load_seeds(sorted(glob.glob(os.path.join(HERE, "runs", f"{tag}_s*.out"))))
        timing, _ = load_seeds([os.path.join(HERE, f"timing_{tag}.out")])
        print(f"[{tag}] seeds={seeds}")
        check_timing_run(tag, data, seeds, timing)
        write_data_files(tag, data)
        print_summary(tag, data, timing)
        data_by_tag[tag], timing_by_tag[tag] = data, timing
    write_table(data_by_tag, timing_by_tag)


if __name__ == "__main__":
    main()
