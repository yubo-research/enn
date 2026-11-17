import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = [
    ("0260", "def", "unbuffered default (0260)", "#d62728", "--"),
    ("0260", "g10", "unbuffered graph sweep, 10 lnN (0260)", "#2ca02c", "--"),
    ("0260", "main", "main, brute force (0260)", "#7f7f7f", ":"),
    ("0261", "b", "buffers, default walk", "#ff7f0e", "-"),
    ("0261", "bg10n", "buffers + graph walk, no sweep", "#9467bd", "-"),
    ("0261", "bg10f", "buffers + Lloyd at flush (new)", "#1f77b4", "-"),
]
PANELS = [
    ("add_us_row", "add time (us / row)"),
    ("q_us_med", "posterior query, median (us)"),
    ("wr_b_row", "bytes written per added row"),
    ("q_rd_ops", "disk read requests per query"),
]


def load(p):
    return [json.loads(x) for x in open(p)] if p.exists() else []


def main():
    reports = Path(sys.argv[1])
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    for (key, label), ax in zip(PANELS, axes.flat):
        for rep, arm, lab, col, ls in ARMS:
            rs = load(
                reports / f"report-{rep}-data" / "cap256" / f"capq_D10_256M_{arm}.jsonl"
            )
            rs = [r for r in rs if r["n"] >= 1000]
            if not rs:
                continue
            ax.plot(
                [r["n"] for r in rs],
                [max(r[key], 0.05) for r in rs],
                ls,
                color=col,
                marker="o",
                ms=3,
                lw=1.4,
                label=lab,
            )
        ax.axvline(3e6, color="k", lw=0.6, alpha=0.4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("N (rows)")
        ax.set_title(label, fontsize=9)
        ax.grid(alpha=0.3, which="both", lw=0.4)
    axes[0, 0].legend(fontsize=6.5, loc="upper left")
    fig.suptitle(
        "256 MB cap, D=10, 1 CPU, EBS; row file 0.9x cap at N=3e6 (vertical line)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(sys.argv[2])


if __name__ == "__main__":
    main()
