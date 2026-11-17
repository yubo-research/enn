import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARMS = [
    ("g10", "unbuffered graph sweep", "#2ca02c"),
    ("bg10n", "buffers, no sweep", "#9467bd"),
    ("bg10f", "buffers + Lloyd at flush", "#1f77b4"),
    ("bg10fc", "... + buffer rows charged", "#17becf"),
    ("b", "buffers, default walk", "#ff7f0e"),
]
PANELS = [
    ("rows_q", "rows scanned / query"),
    ("internal_q", "internal nodes expanded / query"),
    ("q_us", "query time in RAM (us)"),
]


def main():
    with open(sys.argv[1]) as fh:
        rs = [json.loads(x) for x in fh]
    got = {(o["arm"], o["n"]): o for o in rs}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9))
    ns = [100000, 1000000]
    w = 0.16
    for ax, (key, title) in zip(axes, PANELS):
        for i, (arm, lab, col) in enumerate(ARMS):
            ys = [got[(arm, n)][key] if (arm, n) in got else np.nan for n in ns]
            ax.bar(np.arange(2) + (i - 2) * w, ys, w, color=col, label=lab)
        ax.set_xticks(range(2), ["N=1e5", "N=1e6"])
        ax.set_yscale("log")
        ax.set_title(title, fontsize=8.5)
        ax.grid(alpha=0.3, axis="y", which="both", lw=0.4)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, fontsize=6.5, loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(sys.argv[2])


if __name__ == "__main__":
    main()
