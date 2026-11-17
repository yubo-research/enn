import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

R = Path(__file__).resolve().parents[2] / "reports"
D66 = R / "report-0266-data"
MAIN = R / "report-0246-data/score/main.jsonl"
COL = {
    "clean": "#52514e",
    "def": "#2a78d6",
    "b": "#eb6834",
    "g10": "#1baf7a",
    "bg10f": "#eda100",
}
MRK = {"clean": "x", "def": "o", "b": "s", "g10": "^", "bg10f": "D"}
LAB = {
    "def": "default (tree walk)",
    "b": "b: every-level buffers",
    "g10": "g10: graph Lloyd + walk",
    "bg10f": "bg10f: buffers + graph",
}
ARMS = {"1": "def", "2": "b", "3": "g10", "4": "bg10f"}


def load(p):
    return {
        (o["D"], o["n"], o["seed"]): o
        for o in map(json.loads, Path(p).read_text().splitlines())
    }


def quality(ax):
    main = load(MAIN)
    rows = []
    for k, arm in ARMS.items():
        a = load(D66 / f"qual/d10_1e6/r{k}.jsonl")
        ks = sorted(x for x in a if x in main)
        d = np.array([a[x]["loglik"] - main[x]["loglik"] for x in ks])
        rows.append(
            (
                arm,
                d.mean(),
                d.std(ddof=1) / np.sqrt(len(d)),
                stats.ttest_1samp(d, 0).pvalue,
                len(d),
            )
        )
    for i, (arm, m, se, p, n) in enumerate(rows):
        ax.errorbar(
            i, m, yerr=1.96 * se, fmt=MRK[arm], color=COL[arm], ms=8, capsize=4, lw=2
        )
        ax.annotate(
            f"p={p:.2f}\nn={n}",
            (i, m + 1.96 * se),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=7,
        )
    ax.axhline(0, color="#52514e", lw=0.8)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.35 * (hi - lo))
    ax.set_xticks(range(len(rows)), [r[0] for r in rows])
    ax.set_ylabel("loglik minus main (95% CI)")
    ax.set_title("(a) quality, D=10, N=1e6", fontsize=9, loc="left")
    return rows


def capped(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    return np.array(
        [
            (r["n"], r["q_us_med"] / 1000, r["add_us_row"])
            for r in rows
            if r["n"] >= 1000
        ]
    )


def curves(ax_q, ax_a, files, title):
    for arm, f in files.items():
        if not Path(f).exists():
            continue
        c = capped(f)
        if len(c) == 0:
            continue
        ax_q.loglog(
            c[:, 0], c[:, 1], marker=MRK[arm], color=COL[arm], lw=2, ms=5, label=arm
        )
        ax_a.loglog(
            c[1:, 0], c[1:, 2], marker=MRK[arm], color=COL[arm], lw=2, ms=5, label=arm
        )
    for ax, y in ((ax_q, "query ms (median)"), (ax_a, "add us / row")):
        ax.set_xlabel("N rows")
        ax.set_ylabel(y)
        ax.grid(True, which="major", color="#e5e4e0", lw=0.5)
    ax_q.set_title(title, fontsize=9, loc="left")


def main(out):
    fig, axs = plt.subplots(3, 2, figsize=(7.0, 8.4))
    quality(axs[0, 0])
    sizes = json.loads((D66 / "sizes.json").read_text())
    ax = axs[0, 1]
    names = list(sizes)
    before = [sizes[n][0] for n in names]
    after = [sizes[n][1] for n in names]
    y = np.arange(len(names))
    ax.barh(y + 0.2, before, 0.38, color="#86b6ef", label="before (6414a08)")
    ax.barh(y - 0.2, after, 0.38, color="#2a78d6", label="after")
    for i, (b, a) in enumerate(zip(before, after)):
        ax.annotate(
            f"{b}→{a}",
            (max(b, a), i),
            textcoords="offset points",
            xytext=(3, -3),
            fontsize=7,
        )
    ax.set_yticks(y, names, fontsize=7)
    ax.set_xscale("log")
    ax.set_xlim(right=max(before) * 30)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("(b) what the tidy-up removed", fontsize=9, loc="left")
    d10 = {
        "def": R / "report-0260-data/cap256/capq_D10_256M_def.jsonl",
        "g10": R / "report-0260-data/cap256/capq_D10_256M_g10.jsonl",
        "b": R / "report-0261-data/cap256/capq_D10_256M_b.jsonl",
        "bg10f": R / "report-0261-data/cap256/capq_D10_256M_bg10f.jsonl",
    }
    d100 = {
        a: D66 / f"cap256/capq_D100_256M_{a}.jsonl"
        for a in ("def", "b", "g10", "clean")
    }
    curves(axs[1, 0], axs[1, 1], d10, "(c,d) 256 MB cap, D=10 (0260/0261 data)")
    curves(
        axs[2, 0], axs[2, 1], d100, "(e,f) 256 MB cap, D=100 (this report, HEAD build)"
    )
    for a in axs[1:, :].ravel():
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out)


if __name__ == "__main__":
    main(sys.argv[1])
