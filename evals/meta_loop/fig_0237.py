import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#222222", "#777777"


def arm(d, a, reps=("1", "2")):
    out = {}
    for r in reps:
        p = Path(d) / f"c4_D100_1G_a{a}_r{r}.jsonl"
        for o in map(json.loads, p.read_text().splitlines()):
            out.setdefault(o["n"], []).extend(np.array(o["cold"]) / 1e3)
    ns = sorted(out)
    return np.array(ns), np.array([np.median(out[n]) for n in ns])


def ratio(d):
    n, a = arm(d, 0)
    _, b = arm(d, 1)
    return n, b / a


def ram(path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines()]
    by = {(r["advise"], r["seed"]): r for r in rows}
    seeds = sorted({r["seed"] for r in rows})
    res = {}
    for k in ("add_us_row", "q_us_med"):
        x = np.array([by[("off", s)][k] / by[("base", s)][k] for s in seeds])
        res[k] = (x.mean(), x.std(ddof=1) / np.sqrt(len(x)))
    return res


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color="#e6e6e6", lw=0.6)
    ax.tick_params(colors=INK, labelsize=7)


def cold_panel(ax, d, title):
    for a, c, lab in ((0, BLUE, "8-bit (default)"), (1, ORANGE, "4-bit (ENN_CODE4=1)")):
        n, y = arm(d, a)
        ax.plot(n, y, color=c, lw=2, marker="o", ms=4, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N (rows)", fontsize=8)
    ax.set_ylabel("cold query, ms (median)", fontsize=8)
    ax.set_title(title, fontsize=9, loc="left")
    style(ax)


def ratio_panel(ax, root):
    for d, c, lab in (("v4", BLUE, "uniform"), ("v4m5", ORANGE, "5-dim subspace")):
        n, r = ratio(root / d)
        ax.plot(n, r, color=c, lw=2, marker="o", ms=4, label=lab)
    ax.axhline(1.0, color=MUTED, lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("N (rows)", fontsize=8)
    ax.set_ylabel("cold query, 4-bit / 8-bit", fontsize=8)
    ax.set_title("(c) 4-bit helps only at large N", fontsize=9, loc="left")
    ax.legend(fontsize=7, frameon=False)
    style(ax)


def ram_panel(ax, root):
    labels, vals, errs = [], [], []
    for D in (10, 100):
        r = ram(root / "ram6" / f"ram_D{D}.jsonl")
        for k, nm in (("add_us_row", "add"), ("q_us_med", "query")):
            labels.append(f"D={D}\n{nm}")
            vals.append(r[k][0])
            errs.append(r[k][1])
    x = np.arange(len(vals))
    ax.bar(
        x, vals, yerr=errs, color=BLUE, width=0.6, edgecolor="white", lw=2, capsize=3
    )
    for i, v in enumerate(vals):
        ax.text(i, 0.63, f"{v:.2f}", ha="center", fontsize=8, color="white")
    ax.axhline(1.0, color=MUTED, lw=1, ls="--")
    ax.set_xticks(x, labels, fontsize=7)
    ax.set_ylim(0.6, 1.15)
    ax.set_ylabel("in-RAM time, head / base", fontsize=8)
    ax.set_title("(d) default path vs b2d87fa (8 seeds)", fontsize=9, loc="left")
    style(ax)


def main():
    root = Path(sys.argv[1])
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.4))
    cold_panel(axs[0, 0], root / "v4", "(a) D=100, uniform x (no skill)")
    axs[0, 0].legend(fontsize=7, frameon=False)
    cold_panel(axs[0, 1], root / "v4m5", "(b) D=100, x near a 5-dim subspace")
    ratio_panel(axs[1, 0], root)
    ram_panel(axs[1, 1], root)
    fig.tight_layout()
    fig.savefig(root / "fig_0237.pdf")
    fig.savefig(root / "fig_0237.png", dpi=150)


if __name__ == "__main__":
    main()
