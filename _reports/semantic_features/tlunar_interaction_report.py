"""Summarize paired interventions without inferring correlations among optima."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _reports.semantic_features.eval_tlunar_interactions import ROOT, mixed_difference

LABELS = (
    "0 target angle: position",
    "1 target angle: velocity",
    "2 target angle: clip",
    "3 target hover: distance",
    "4 flight angle: error",
    "5 flight angle: rate",
    "6 flight hover: error",
    "7 flight hover: rate",
    "8 contact angle: bias",
    "9 contact hover: rate",
    "10 main hover: deadband",
    "11 side angle: deadband",
)
LOCAL_PAIRS = ((0, 4), (1, 4), (3, 6))


def interaction_summary(returns, pairs):
    delta = mixed_difference(returns)
    energy = np.mean(delta**2, axis=2)
    rng = np.random.default_rng(20261029)
    indices = rng.integers(0, len(returns), (4000, len(returns)))
    boot = np.sqrt(energy[indices].mean(axis=1))
    intervals = np.quantile(boot, [0.025, 0.975], axis=0)
    rows = []
    for p, (i, j) in enumerate(pairs):
        rows.append(
            {
                "i": int(i),
                "j": int(j),
                "rms_contrast": float(np.sqrt(energy[:, p].mean())),
                "ci_low": float(intervals[0, p]),
                "ci_high": float(intervals[1, p]),
                "nonzero_fraction": float(np.mean(np.abs(delta[:, p]) > 1e-9)),
            }
        )
    return rows


def heatmap(output, rows):
    matrix = np.full((12, 12), np.nan)
    for row in rows:
        i, j = row["i"], row["j"]
        matrix[i, j] = matrix[j, i] = row["rms_contrast"]
    fig, ax = plt.subplots(figsize=(7.5, 5.7))
    plot = ax.imshow(np.ma.masked_invalid(matrix), cmap="viridis", vmin=0)
    ax.set_xticks(range(12), range(12))
    ax.set_yticks(range(12), LABELS, fontsize=8)
    ax.set_xlabel("Controller weight index")
    ax.set_title(
        "RMS paired return contrast\nLarger means stronger nonadditivity at tested levels",
        fontsize=10,
    )
    fig.colorbar(plot, ax=ax, label="Return units")
    fig.tight_layout()
    fig.savefig(output / "interactions.pdf")
    fig.savefig(output / "interactions.png", dpi=150)
    plt.close(fig)


def summary_table(output, rows):
    ordered = sorted(rows, key=lambda r: -r["rms_contrast"])
    selected = ordered[:5] + [
        r for r in rows if (r["i"], r["j"]) in LOCAL_PAIRS and r not in ordered[:5]
    ]
    lines = []
    for row in selected:
        lines.append(
            f"{row['i']}, {row['j']} & {row['rms_contrast']:.1f} & "
            f"[{row['ci_low']:.1f}, {row['ci_high']:.1f}] & "
            f"{100 * row['nonzero_fraction']:.0f}" + r"\% \\"
        )
    (output / "pair_table.tex").write_text(
        "\n".join(lines) + "\n\\bottomrule\\end{tabular}\n"
    )
    return ordered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "tlunar_interactions")
    args = parser.parse_args()
    with np.load(args.output / "interactions.npz") as data:
        rows = interaction_summary(data["returns"], data["pairs"])
    with (args.output / "pairs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    heatmap(args.output, rows)
    ordered = summary_table(args.output, rows)
    result = {
        "top_pairs": ordered[:5],
        "local_product_pairs": [r for r in rows if (r["i"], r["j"]) in LOCAL_PAIRS],
        "median_rms": float(np.median([r["rms_contrast"] for r in rows])),
        "min_rms": float(min(r["rms_contrast"] for r in rows)),
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
