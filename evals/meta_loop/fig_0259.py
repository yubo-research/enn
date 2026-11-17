import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REP = Path(__file__).resolve().parents[2] / "reports"
D59 = REP / "report-0259-data"
D58 = REP / "report-0258-data" / "gw"
C = {
    "new": "#2a78d6",
    "old": "#eb6834",
    "tree": "#1baf7a",
    "ideal": "#eda100",
    "prune": "#6250d6",
}


def rows(paths, arm_order):
    out = {}
    for p in paths:
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            o = json.loads(line)
            if (o["order"], o["arm"] != "oracle") == arm_order:
                out[o["N"]] = o
    return [out[n] for n in sorted(out)]


def series():
    gw = "gw_gw2e10p0.jsonl"
    new = [D59 / "gw_f" / gw]
    old = [D59 / "gw_base1" / gw, D59 / "gw_base3" / gw]
    old58 = [D58 / "gw_gw2e10.jsonl"]
    tree = [D58 / "gw_gw0e10.jsonl"]
    return {
        "new": (
            "merged walk, on-disk coded lists + prune (0259)",
            rows(new, ("walk", True)),
        ),
        "old": (
            "merged walk, neighbour-page reads (HEAD)",
            rows(old58 + old, ("walk", True)),
        ),
        "old58": ("merged walk, 0258 record", rows(old58, ("walk", True))),
        "tree": ("tree walk (0258)", rows(tree, ("walk", True))),
        "ideal": ("ideal centroid order", rows(new, ("ideal", True))),
    }


def loglog(ax, s, field, keys, ylabel, title, *, slopes=True, loc="upper left"):
    for k in keys:
        name, rs = s[k]
        if not rs:
            continue
        x = np.array([r["N"] for r in rs], float)
        y = np.array([r[field] for r in rs], float)
        if slopes and len(x) >= 3:
            name += f" ($N^{{{np.polyfit(np.log(x), np.log(y), 1)[0]:.2f}}}$)"
        ax.plot(x, y, "-o", color=C.get(k, "#888888"), lw=2, ms=6, label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N (rows)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.25, which="both", lw=0.5)
    ax.legend(fontsize=6.5, frameon=False, loc=loc)


def arm_q(st, case, arm):
    r = next((r for r in st if r["case"] == case and r["arm"] == arm), None)
    return r["query_s"] if r else np.nan


def timing(ax, stats_path, prune_path):
    if not stats_path.exists():
        ax.text(0.5, 0.5, "timing pending", ha="center", transform=ax.transAxes)
        return
    st = json.loads(stats_path.read_text())
    pr = json.loads(prune_path.read_text()) if prune_path.exists() else []
    arms = [
        (st, "10", "tree walk", "tree"),
        (st, "212", "merged, HEAD offers", "old"),
        (st, "12", "merged, coded lists", "new"),
        (pr, "13", "coded lists + offer prune", "prune"),
    ]
    cases = sorted({r["case"] for r in st})
    w = 0.2
    for j, (src, arm, lab, col) in enumerate(arms):
        ys = [arm_q(src, c, arm) for c in cases]
        xs = np.arange(len(cases)) + (j - 1.5) * w
        ax.bar(xs, ys, w - 0.02, color=C[col], label=lab)
        for xx, yy in zip(xs, ys):
            if np.isfinite(yy):
                ax.text(xx, yy, f"{yy:.2f}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(np.arange(len(cases)))
    ax.set_xticklabels([c.replace("d10_", "N=") for c in cases])
    ax.set_ylabel("in-RAM query time (ms / query)")
    ax.set_title("(c) In-RAM time, default budget 100 lnN", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    ax.grid(alpha=0.25, axis="y", lw=0.5)


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else D59 / "fig_0259.pdf"
    s = series()
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.3))
    loglog(
        axs[0],
        s,
        "ltr_p50",
        ["tree", "old", "new", "ideal"],
        "median leaves to last true neighbour",
        "(a) Visit order (D=10, 300 queries)",
    )
    loglog(
        axs[1],
        s,
        "walk_leaves_mean",
        ["tree", "old", "new"],
        "leaves visited per query (mean)",
        "(b) Walk length at 100 lnN budget",
        slopes=False,
        loc="lower right",
    )
    timing(axs[2], D59 / "qual" / "stats.json", D59 / "prune" / "stats.json")
    fig.tight_layout()
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=130)
    print(out)


if __name__ == "__main__":
    main()
