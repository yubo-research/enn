"""Build data files and tables for report-metric_learning.tex from three sets of eval runs.

- ../metric_12d/seeds10.out, ranges10.out: `./ops/evaluate.py run short/metric_12d|metric_ranges`
  (10 seeds; flat, flat_scale_x, bpann_disk, bpann_disk_metric_learning, bpann_disk_auto)
- scale_x_12d.out, scale_x_ranges.out: `run_scale_x.py 12d|ranges` (bpann_disk_scale_x)
- stream_runs/{12d,ranges}_s*.out: `run_streaming.py` (the five streaming metric models, with
  bpann_disk and bpann_disk_metric_learning rerun as controls and as same-run timing references)
- stream_runs/{12d,ranges}_row_s*.out: `run_streaming.py TAG SEED 2 bpann_disk_spsa_row bpann_disk_spsa_sobol_row`
  (the SPSA learner called once per row)
- timing_{12d,ranges}.out, timing_row_{12d,ranges}.out: the same with seed 0 only, run alone (cost table)

Every value is a mean ± standard error over seeds, computed from the per-seed lines. Reruns of a
model are checked against the earlier run (same seeds, same data) and must agree.
Usage: python reports/metric_learning/make_report.py
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
EARLIER = (
    "flat",
    "flat_scale_x",
    "bpann_disk",
    "bpann_disk_metric_learning",
    "bpann_disk_auto",
)
STREAM = (
    "bpann_disk_ml_reservoir",
    "bpann_disk_sobol",
    "bpann_disk_corr",
    "bpann_disk_spsa",
    "bpann_disk_spsa_sobol",
)
CONTROLS = ("bpann_disk", "bpann_disk_metric_learning")
ROW = ("bpann_disk_spsa_row", "bpann_disk_spsa_sobol_row")
RUNS = {
    "12d": [os.path.join(HERE, "..", "metric_12d", "seeds10.out")],
    "ranges": [os.path.join(HERE, "ranges10.out")],
}
MODELS = [
    ("flat", "\\texttt{NONE}"),
    ("bpann_disk_scale_x", "\\texttt{SCALE\\_X}"),
    ("bpann_disk_metric_learning", "L-BFGS-B, fresh subsample"),
    ("bpann_disk_ml_reservoir", "L-BFGS-B, reservoir"),
    ("bpann_disk_sobol", "Sobol weights"),
    ("bpann_disk_corr", "Correlation weights"),
    ("bpann_disk_spsa_row", "SPSA from \\texttt{SCALE\\_X}"),
    ("bpann_disk_spsa_sobol_row", "SPSA from Sobol"),
    ("bpann_disk_spsa", "\\quad batched, from \\texttt{SCALE\\_X}"),
    ("bpann_disk_spsa_sobol", "\\quad batched, from Sobol"),
    ("bpann_disk_auto", "\\texttt{AUTO} (L-BFGS-B)"),
]
COST_MODELS = ("bpann_disk", "bpann_disk_metric_learning", *STREAM)
ROW_COST_MODELS = ("bpann_disk", *ROW)
METRICS = ("loglik", "nrmse", "add", "q")
PAIRS = [
    ("bpann_disk_ml_reservoir", "bpann_disk_metric_learning"),
    ("bpann_disk_sobol", "bpann_disk_metric_learning"),
    ("bpann_disk_corr", "bpann_disk_metric_learning"),
    ("bpann_disk_spsa", "bpann_disk_metric_learning"),
    ("bpann_disk_spsa_sobol", "bpann_disk_metric_learning"),
    ("bpann_disk_spsa_sobol", "bpann_disk_sobol"),
    ("bpann_disk_spsa_row", "bpann_disk_spsa"),
    ("bpann_disk_spsa_sobol_row", "bpann_disk_spsa_sobol"),
    ("bpann_disk_spsa_row", "bpann_disk_metric_learning"),
    ("bpann_disk_spsa_sobol_row", "bpann_disk_sobol"),
    ("bpann_disk_spsa_row", "bpann_disk_scale_x"),
    ("bpann_disk_sobol", "bpann_disk_corr"),
    ("bpann_disk_sobol", "bpann_disk_scale_x"),
    ("bpann_disk_spsa", "bpann_disk_scale_x"),
    ("bpann_disk_sobol", "flat"),
]


def mean_se(values):
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size))


def load_seeds(paths, num_models):
    """{(model, n): array of shape (num_seeds, 4)} with columns loglik, nrmse, add_s, query_s."""
    text = "".join(open(p).read() for p in paths)
    rows = {}
    for m in SEED_LINE.finditer(text):
        seed, model, n, *vals = m.groups()
        rows.setdefault((model, int(n)), {})[int(seed)] = [float(v) for v in vals]
    seeds = sorted(next(iter(rows.values())))
    if any(sorted(r) != seeds for r in rows.values()) or len(rows) != 11 * num_models:
        raise ValueError(
            f"{paths}: every (model, n) must have the same seeds; is the run complete?"
        )
    return {k: np.array([r[s] for s in seeds]) for k, r in rows.items()}, len(seeds)


def ns_of(data, model):
    return sorted(n for m, n in data if m == model)


def check_rerun(tag, data, new, model):
    """Both runs start at seed 0, so ``new``'s seeds are the first rows of ``data``'s."""
    diff = max(
        float(
            np.abs(
                new[(model, n)][:, :2] - data[(model, n)][: len(new[(model, n)]), :2]
            ).max()
        )
        for n in ns_of(new, model)
    )
    print(
        f"[{tag}] {model} rerun vs earlier run: max |diff| of loglik and nrmse = {diff:.2g}"
    )


def load_all(tag):
    """Earlier models, bpann_disk_scale_x from its run, and the streaming models; plus the streaming run itself."""
    data, num_seeds = load_seeds(RUNS[tag], len(EARLIER))
    scale_x, _ = load_seeds([os.path.join(HERE, f"scale_x_{tag}.out")], 2)
    stream_paths = sorted(glob.glob(os.path.join(HERE, "stream_runs", f"{tag}_s*.out")))
    stream, stream_seeds = load_seeds(stream_paths, len(STREAM) + len(CONTROLS))
    if stream_seeds != num_seeds:
        raise ValueError(
            f"{tag}: {stream_seeds} seeds in the streaming run, {num_seeds} in the earlier one"
        )
    check_rerun(tag, data, scale_x, "bpann_disk")
    for model in CONTROLS:
        check_rerun(tag, data, stream, model)
    data.update({k: v for k, v in scale_x.items() if k[0] == "bpann_disk_scale_x"})
    data.update({k: v for k, v in stream.items() if k[0] in STREAM})
    row, row_seeds = load_seeds(
        sorted(glob.glob(os.path.join(HERE, "stream_runs", f"{tag}_row_s*.out"))),
        len(ROW),
    )
    if row_seeds != num_seeds:
        raise ValueError(
            f"{tag}: {row_seeds} seeds in the per-row run, {num_seeds} in the earlier one"
        )
    data.update(row)
    return data, stream, num_seeds


def write_data_files(tag, data):
    header = " ".join(f"{c} {c}_se" for c in METRICS)
    for model, _ in MODELS:
        lines = [f"n {header}"]
        for n in ns_of(data, model):
            stats = (mean_se(data[(model, n)][:, i]) for i in range(len(METRICS)))
            lines.append(f"{n} " + " ".join(f"{mu:.5g} {se:.5g}" for mu, se in stats))
        with open(os.path.join(HERE, f"d_{tag}_{model}.dat"), "w") as f:
            f.write("\n".join(lines) + "\n")


def pm(stat, fmt):
    return f"${stat[0]:{fmt}} \\pm {stat[1]:{fmt.lstrip('+')}}$"


def accuracy_cells(data, model):
    cells = [pm(mean_se(data[(model, n)][:, 0]), "+.2f") for n in (1000, 1000000)]
    return cells + [pm(mean_se(data[(model, 1000000)][:, 1]), ".2f")]


def write_accuracy_table(data_by_tag):
    """One row per model: loglik at 1e3 and 1e6 and nrmse at 1e6, for each problem."""
    lines = []
    for model, label in MODELS:
        cells = [
            c
            for tag in ("12d", "ranges")
            for c in accuracy_cells(data_by_tag[tag], model)
        ]
        lines.append(f"{label} & " + " & ".join(cells) + "\\\\")
    with open(os.path.join(HERE, "t_both.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def cost_row(stream, model):
    ns = ns_of(stream, model)
    total_add = sum(stream[(model, n)][:, 2] for n in ns)
    extra = total_add - sum(stream[("bpann_disk", n)][:, 2] for n in ns)
    return (
        np.median(total_add),
        np.median(extra),
        np.median(stream[(model, ns[-1])][:, 3]),
    )


def write_cost_table(stream_by_tag, row_by_tag):
    labels = dict(MODELS) | {"bpann_disk": "\\texttt{NONE}"}
    lines = []
    for model in COST_MODELS[:5] + ROW + COST_MODELS[5:]:
        cells = []
        for tag in ("12d", "ranges"):
            total, extra, query = cost_row(
                (row_by_tag if model in ROW else stream_by_tag)[tag], model
            )
            cells.append(f"${total:.1f}$ & ${extra:+.1f}$ & ${query:.1f}$")
        lines.append(f"{labels[model]} & " + " & ".join(cells) + "\\\\")
    with open(os.path.join(HERE, "t_cost.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def print_summary(tag, data, stream):
    """Per-checkpoint numbers quoted in the text, including paired (same-seed) differences."""
    for n in ns_of(data, "flat"):
        parts = [
            "{}={:+.3f}±{:.3f}".format(m, *mean_se(data[(m, n)][:, 0]))
            for m, _ in MODELS
        ]
        diffs = [
            "{}-{}={:+.3f}±{:.3f}#{}".format(
                a,
                b,
                *mean_se(data[(a, n)][:, 0] - data[(b, n)][:, 0]),
                int(np.sum(data[(a, n)][:, 0] > data[(b, n)][:, 0])),
            )
            for a, b in PAIRS
        ]
        print(f"[{tag}] n={n} " + " ".join(parts + diffs))
    for model, _ in MODELS:
        print(
            f"[{tag}] {model}: loglik per seed at 1e6 {np.round(data[(model, 1000000)][:, 0], 2).tolist()}"
        )
    for model in COST_MODELS:
        total, extra, query = cost_row(stream, model)
        print(
            f"[{tag}] contended run {model}: median total add {total:.2f} s (+{extra:.2f} over NONE), query {query:.2f} s"
        )


def print_summary_cost(tag, timing, models):
    """Uncontended single-seed costs: extra add time per fit (11 checkpoints) and per streamed row."""
    num_fits, num_rows = (
        len(ns_of(timing, "bpann_disk")),
        max(ns_of(timing, "bpann_disk")),
    )
    for model in models:
        total, extra, query = cost_row(timing, model)
        print(
            f"[{tag}] uncontended {model}: total add {total:.2f} s, extra {extra:+.2f} s "
            f"= {extra / num_fits:.3f} s per checkpoint = {1e6 * extra / num_rows:.1f} us per row, query {query:.2f} s"
        )


def main():
    timing_by_tag, row_timing_by_tag, data_by_tag = {}, {}, {}
    for tag in RUNS:
        data, stream, num_seeds = load_all(tag)
        timing, _ = load_seeds(
            [os.path.join(HERE, f"timing_{tag}.out")], len(COST_MODELS)
        )
        row_timing, _ = load_seeds(
            [os.path.join(HERE, f"timing_row_{tag}.out")], len(ROW_COST_MODELS)
        )
        check_rerun(tag, data, timing, "bpann_disk_sobol")
        check_rerun(tag, data, row_timing, "bpann_disk_spsa_row")
        timing_by_tag[tag], row_timing_by_tag[tag], data_by_tag[tag] = (
            timing,
            row_timing,
            data,
        )
        print(f"[{tag}] seeds={num_seeds}")
        write_data_files(tag, data)
        print_summary(tag, data, stream)
        print_summary_cost(tag, timing, COST_MODELS)
        print_summary_cost(tag, row_timing, ROW_COST_MODELS)
    write_accuracy_table(data_by_tag)
    write_cost_table(timing_by_tag, row_timing_by_tag)


if __name__ == "__main__":
    main()
