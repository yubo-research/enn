import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

D, OUT = Path(sys.argv[1]), sys.argv[2]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
C = {"gw2": "#2a78d6", "gw0": "#eb6834", "gw1": "#1baf7a"}
NAME = {
    "gw2": "graph walk, merged",
    "gw0": "current tree walk",
    "gw1": "graph walk, graph-first",
}


def load():
    recs = []
    for f in D.glob("gw_*.jsonl"):
        recs += [json.loads(line) for line in f.read_text().splitlines()]
    return [r for r in recs if r["seed"] == 0]


def series(recs, arm, order, key):
    pts = sorted(
        (r["N"], r[key]) for r in recs if r["arm"] == arm and r["order"] == order
    )
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def style(ax, title, ylabel):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.set_title(title, loc="left", fontsize=9, color=INK)
    ax.set_xlabel("N (rows)", fontsize=8, color=INK2)
    ax.set_ylabel(ylabel, fontsize=8, color=INK2)
    ax.tick_params(labelsize=7, colors=INK2)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(
        matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{v:g}" if str(int(v))[0] in "25" else ""
        )
    )
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def panel_order(ax, recs):
    for k in ("gw0", "gw1", "gw2"):
        x, y = series(recs, f"{k}e10", "walk", "ltr_p50")
        if len(x):
            ax.plot(x, y, "-o", color=C[k], lw=2, ms=4, label=NAME[k])
            ax.annotate(
                f"{y[-1]:.0f}",
                (x[-1], y[-1]),
                xytext=(4, 0),
                textcoords="offset points",
                fontsize=7,
                color=INK,
                va="center",
            )
    x, y = series(recs, "gw2e10", "ideal", "ltr_p50")
    ax.plot(x, y, "--", color=INK2, lw=1.2, label="ideal centroid order")
    x, y = series(recs, "oracle", "ideal", "ltr_p50")
    ax.plot(x, y, ":", color=INK2, lw=1.2, label="oracle (true leaves only)")
    ax.plot([1e6], [36], marker="_", ms=14, mew=2, color=INK)
    ax.annotate(
        "reject if > 36",
        (1e6, 36),
        xytext=(6, -10),
        textcoords="offset points",
        fontsize=7,
        color=INK,
        ha="left",
    )
    style(ax, "(a) Leaves to the last true neighbour (median)", "leaves")
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")


def panel_cost(ax, recs):
    shade = {"0": "#184f95", "30": "#2a78d6", "10": "#5598e7", "3": "#9ec5f4"}
    arms = sorted(
        {
            r["arm"]
            for r in recs
            if r["arm"].startswith("gw2e10") and r["arm"] != "gw2e10p0"
        }
    )
    for a in arms:
        x, y = series(recs, a, "walk", "walk_leaves_mean")
        _, rec = series(recs, a, "walk", "ltr_miss")
        p = a.split("p")[-1] if "p" in a[5:] else "0"
        lab = "no patience stop" if p == "0" else f"patience {p} lnN"
        ax.plot(
            x,
            y,
            "-o",
            lw=2,
            ms=4,
            color=shade[p],
            label=lab,
            zorder=3 if p == "0" else 2,
        )
        if p != "30":
            ax.annotate(
                f"{lab}: miss {rec[-1]:.3f}",
                (x[-1], y[-1]),
                xytext=(5, -2),
                textcoords="offset points",
                fontsize=6,
                color=INK2,
            )
    x, y = series(recs, "gw0e10", "walk", "walk_leaves_mean")
    ax.plot(x, y, "--", color=C["gw0"], lw=1.2, label="current tree walk (same length)")
    style(ax, "(b) Leaves walked per query, merged walk", "leaves walked")
    ax.set_xlim(7e3, 2.5e7)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")


recs = load()
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
panel_order(axs[0], recs)
panel_cost(axs[1], recs)
fig.tight_layout()
fig.savefig(OUT)
fig.savefig(OUT.replace(".pdf", ".png"), dpi=150)
