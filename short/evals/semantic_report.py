"""Produce report tables and paired uncertainty from the recorded benchmark."""

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
METHODS = ["baseline", "semantic", "shuffled"]
SCENARIOS = ["A_related", "B_independent", "C_permuted"]


def seed_values(rows, scenario, method, metric, count=None):
    return np.array(
        [
            np.mean(
                [
                    float(r[metric])
                    for r in rows
                    if r["scenario"] == scenario
                    and r["method"] == method
                    and int(r["seed"]) == seed
                    and (count is None or int(r["n"]) == count)
                ]
            )
            for seed in range(5)
        ]
    )


def mean_se(values):
    return values.mean(), values.std(ddof=1) / np.sqrt(len(values))


def table_body(lines):
    return "\n".join(lines) + "\n\\bottomrule\\end{tabular}\n"


def metric_cells(rows, scenario, count, metric):
    cells = []
    for method in METHODS:
        mean, se = mean_se(seed_values(rows, scenario, method, metric, count))
        cells.append(f"${mean:.3f} \\pm {se:.3f}$")
    return cells


def tables(rows):
    for metric in ["nrmse", "loglik"]:
        lines = []
        for scenario in SCENARIOS:
            for count in [50, 100, 200, 500]:
                cells = [scenario[0], str(count)]
                cells.extend(metric_cells(rows, scenario, count, metric))
                lines.append(" & ".join(cells) + r" \\")
        (ROOT / f"{metric}_table.tex").write_text(table_body(lines))
    lines = []
    for scenario in SCENARIOS:
        baseline = seed_values(rows, scenario, "baseline", "nrmse")
        for method in METHODS[1:]:
            values = seed_values(rows, scenario, method, "nrmse")
            change = 100 * (values / baseline - 1)
            mean, se = mean_se(change)
            delta, delta_se = mean_se(values - baseline)
            lines.append(
                f"{scenario[0]} & {method} & ${mean:+.2f} \\pm {se:.2f}$ & ${delta:+.3f} \\pm {delta_se:.3f}$"
                + r" \\"
            )
    (ROOT / "paired_table.tex").write_text(table_body(lines))


def plot(rows):
    figure, axes = plt.subplots(2, 3, figsize=(9, 4.8), sharex=True)
    colors = ["#555555", "#0072b2", "#d55e00"]
    for column, scenario in enumerate(SCENARIOS):
        for row, metric in enumerate(["nrmse", "loglik"]):
            for method, color in zip(METHODS, colors):
                results = [
                    mean_se(seed_values(rows, scenario, method, metric, count))
                    for count in [50, 100, 200, 500]
                ]
                axes[row, column].errorbar(
                    [50, 100, 200, 500],
                    [r[0] for r in results],
                    yerr=[r[1] for r in results],
                    color=color,
                    label=method,
                    marker="o",
                    capsize=2,
                    linewidth=1,
                )
            axes[row, column].grid(alpha=0.2)
            axes[row, column].set_xscale("log")
            axes[row, column].set_xticks(
                [50, 100, 200, 500], ["50", "100", "200", "500"]
            )
        axes[0, column].set_title(scenario.replace("_", ": "))
        axes[1, column].set_xlabel("Training observations")
    axes[0, 0].set_ylabel("NRMSE (lower is better)")
    axes[1, 0].set_ylabel("LogLik (higher is better)")
    axes[0, 0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(ROOT / "semantic_results.pdf")
    plt.close(figure)


def main():
    with (ROOT / "semantic_raw.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    tables(rows)
    plot(rows)


if __name__ == "__main__":
    main()
