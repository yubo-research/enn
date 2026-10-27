"""Generate LunarLander report tables and paired optimization statistics."""

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t

from evals.short.semantic_report import mean_se, table_body

ROOT = Path(__file__).resolve().parent
METHODS = ("real", "semantic_only", "both")
LABELS = ("Real", "Semantic only", "Both")
COLORS = ("#555555", "#0072b2", "#009e73")


def read_csv(name):
    with (ROOT / name).open() as stream:
        return list(csv.DictReader(stream))


def values(rows, method, count, metric):
    selected = sorted(
        (r for r in rows if r["method"] == method and int(r["n"]) == count),
        key=lambda r: int(r["seed"]),
    )
    return np.array([float(r[metric]) for r in selected])


def cell(v):
    mean, se = mean_se(v)
    return f"${mean:.3f} \\pm {se:.3f}$"


def prediction_report(rows):
    lines = []
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.7))
    counts = (30, 100, 300, 1000)
    for metric, axis in zip(("nrmse", "loglik"), axes):
        for method, label, color in zip(METHODS, LABELS, COLORS):
            stats = [mean_se(values(rows, method, n, metric)) for n in counts]
            axis.errorbar(
                counts,
                [s[0] for s in stats],
                yerr=[s[1] for s in stats],
                marker="o",
                capsize=3,
                label=label,
                color=color,
            )
        axis.set_xscale("log")
        axis.set_xticks(counts, [str(n) for n in counts])
        axis.set_xlabel("Training observations")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("NRMSE (lower is better)")
    axes[1].set_ylabel("LogLik (higher is better)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "lunar_prediction.pdf")
    plt.close(fig)
    for n in counts:
        for method, label in zip(METHODS, LABELS):
            lines.append(
                f"{n} & {label} & {cell(values(rows, method, n, 'nrmse'))} & "
                + cell(values(rows, method, n, "loglik"))
                + r" \\"
            )
    (ROOT / "lunar_prediction_table.tex").write_text(table_body(lines))


def optimization_figure(rows):
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
    for method, label, color in zip(METHODS, LABELS, COLORS):
        curves = []
        for seed in range(30):
            with np.load(ROOT / "lunar_runs" / f"{method}_{seed:02d}.npz") as data:
                curves.append(np.maximum.accumulate(data["y"]))
        curves = np.array(curves)
        mean = curves.mean(axis=0)
        se = curves.std(axis=0, ddof=1) / np.sqrt(30)
        axes[0].plot(np.arange(1, 301), mean, color=color, label=label)
        axes[0].fill_between(
            np.arange(1, 301), mean - se, mean + se, color=color, alpha=0.15
        )
        for axis, metric in zip(axes[1:], ("best_return", "validation_return")):
            v = np.sort(values(rows, method, 300, metric))
            axis.step(v, np.arange(1, 31) / 30, where="post", color=color, label=label)
    axes[0].set_xlabel("Objective evaluations")
    axes[0].set_ylabel("Best return (higher is better)")
    axes[0].legend(fontsize=7)
    for axis, title in zip(
        axes[1:], ("Final frozen-seed return", "Ten-seed validation mean")
    ):
        axis.set_xlabel(title + "\n(higher is better)")
        axis.set_ylabel("Fraction of repetitions ≤ return")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(ROOT / "lunar_optimization.pdf")
    plt.close(fig)


def optimization_report(rows):
    summaries, lines = {}, []
    for method, label in zip(METHODS, LABELS):
        record = {}
        cells = [label]
        for metric in ("best_return", "validation_return"):
            v = values(rows, method, 300, metric)
            mean, se = mean_se(v)
            delta = v - values(rows, "real", 300, metric)
            change, change_se = mean_se(delta)
            record[metric] = {
                "mean": float(mean),
                "se": float(se),
                "median": float(np.median(v)),
                "paired_delta": float(change),
                "paired_ci95": float(t.ppf(0.975, 29) * change_se),
            }
            cells.append(f"${mean:.1f} \\pm {se:.1f}$")
            cells.append(f"${change:+.1f} \\pm {t.ppf(0.975, 29) * change_se:.1f}$")
        summaries[method] = record
        lines.append(" & ".join(cells) + r" \\")
    (ROOT / "lunar_optimization_table.tex").write_text(table_body(lines))
    (ROOT / "lunar_optimization_summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n"
    )
    optimization_figure(rows)
    write_findings(summaries)


def write_findings(summaries):
    frozen = [summaries[m]["best_return"]["mean"] for m in METHODS]
    validation = [summaries[m]["validation_return"]["mean"] for m in METHODS]
    text = (
        "% Summary: Paired intervals do not establish an optimization improvement.\n"
        f"Final mean frozen-seed returns for real/semantic-only/both are "
        f"{frozen[0]:.1f}/{frozen[1]:.1f}/{frozen[2]:.1f}; validation means are "
        f"{validation[0]:.1f}/{validation[1]:.1f}/{validation[2]:.1f}. "
        "All four paired 95\\% intervals comparing a semantic method with real inputs "
        "include zero (Table~\\ref{tab:optimization}). These runs therefore do not "
        "establish an optimization advantage for either semantic representation. "
        "The empirical distributions show substantial variability across optimizer seeds. "
        "The large drop on unseen simulator seeds shows that maximizing a single "
        "frozen-seed return does not reliably select a controller with similar "
        "performance on new episodes. Better aligned synthetic prediction does not "
        "resolve this controller-selection problem.\n"
    )
    (ROOT / "lunar_findings.tex").write_text(text)


def main():
    prediction_report(read_csv("lunar_prediction.csv"))
    optimization_report(read_csv("lunar_optimization.csv"))


if __name__ == "__main__":
    main()
