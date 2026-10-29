"""Render measured tables and curves for the two-column adaptation report."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullLocator

ROOT = Path("_reports/adapt")
DATA = ROOT / "confirmatory"
SETTINGS = (
    ("xx", "aligned"),
    ("xx", "misleading"),
    ("yy", "aligned"),
    ("yy", "opposing"),
    ("yy", "independent"),
)


def estimate(values):
    values = np.array(values)
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def selected(rows, kind, case, method, count=300):
    return [
        r
        for r in rows
        if r["kind"] == kind
        and r["case"] == case
        and r["method"] == method
        and r["n"] == count
    ]


def tex_stat(values, digits=3):
    mean, se = estimate(values)
    return f"${mean:.{digits}f} \\pm {se:.{digits}f}$"


def tables(rows):
    lines, paired, ablation = [], [], []
    names = {"raw": "Raw ENN", "fixed": "Fixed prior", "adaptive": "Adaptive"}
    for kind, case in SETTINGS:
        for method, method_label in names.items():
            values = selected(rows, kind, case, method)
            label = ("xx/xy" if kind == "xx" else "yy") + ": " + case
            fields = [
                tex_stat([r[key] for r in values])
                for key in ("nrmse", "loglik", "coverage")
            ]
            fields.append(tex_stat([1000 * r["update_s"] for r in values], 2))
            lines.append(" & ".join([label, method_label, *fields]) + " \\\\")
        baseline = {r["seed"]: r for r in selected(rows, kind, case, "fixed")}
        differences = [
            r["nrmse"] - baseline[r["seed"]]["nrmse"]
            for r in selected(rows, kind, case, "adaptive")
        ]
        mean, se = estimate(differences)
        paired.append(
            {
                "kind": kind,
                "case": case,
                "difference": mean,
                "ci95": [mean - 2.056 * se, mean + 2.056 * se],
            }
        )
    for case in ("aligned", "opposing", "independent"):
        fields = [
            tex_stat([r["nrmse"] for r in selected(rows, "yy", case, method)])
            for method in (
                "tuned_fixed",
                "adaptive_inputs",
                "adaptive_outputs",
                "adaptive",
                "adaptive_linear",
            )
        ]
        ablation.append(" & ".join([case, *fields]) + " \\\\")
    (ROOT / "table.tex").write_text("\n".join(lines) + "\n\\bottomrule\n")
    (ROOT / "ablation.tex").write_text("\n".join(ablation) + "\n\\bottomrule\n")
    (ROOT / "paired.json").write_text(json.dumps(paired, indent=2) + "\n")


def curves(rows):
    plt.rcParams.update(
        {"font.size": 8, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 3.8), constrained_layout=True)
    for ax, (kind, case) in zip(axes.flat, SETTINGS):
        for method, label in (
            ("raw", "Raw"),
            ("fixed", "Fixed"),
            ("adaptive", "Adaptive"),
        ):
            stats = [
                estimate([r["nrmse"] for r in selected(rows, kind, case, method, n)])
                for n in (30, 100, 300)
            ]
            ax.errorbar(
                (30, 100, 300),
                [s[0] for s in stats],
                yerr=[s[1] for s in stats],
                marker="o",
                ms=3,
                label=label,
            )
        ax.set(
            xscale="log",
            xlabel="Observations",
            ylabel="NRMSE (lower is better)",
            title=f"{'xx/xy' if kind == 'xx' else kind}: {case}",
        )
        ax.set_xticks((30, 100, 300), ("30", "100", "300"))
        ax.xaxis.set_minor_locator(NullLocator())
    saturation = json.loads((ROOT / "saturation.json").read_text())
    ax = axes.flat[-1]
    for key, label in (("retained", "Retained replay rows"),):
        counts = sorted({r["n"] for r in saturation})
        means = [np.mean([r[key] for r in saturation if r["n"] == n]) for n in counts]
        ax.plot(counts, means, label=label)
    ax.set(
        xlabel="Observations seen",
        ylabel="Rows retained",
        title="Replay bound (capacity 300)",
        ylim=(0, 350),
    )
    axes.flat[0].legend(frameon=False, fontsize=7)
    fig.savefig(ROOT / "curves.pdf")
    plt.close(fig)


def main():
    rows = []
    for kind in ("xx", "xy", "yy"):
        rows.extend(json.loads((DATA / f"{kind}.json").read_text()))
    tables(rows)
    curves(rows)


if __name__ == "__main__":
    main()
