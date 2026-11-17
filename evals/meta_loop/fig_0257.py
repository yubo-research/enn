import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
REF = ROOT.parent / "report-0256-data" / "growth"
LINE = re.compile(r"^(\S+)\s+(walk|ideal)\s+ltr p50/p90 (\d+)/(\d+)")
DONE = re.compile(r"build (\d+)s")
NS = {"1e4": 1e4, "1e5": 1e5, "1e6": 1e6, "3e6": 3e6}
INK, MUTED = "#0b0b0b", "#52514e"
SERIES = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
]

ARMS = [
    ("default tree", "ref", "r64", "r64"),
    ("subtree pass every 3 adds", "every", "e3", "e3"),
    ("tree-searched hood, k=3, 100 pages", "lloyd", "g3p100", "g3p100"),
    ("exact hood, k=30", "lloyd", "g30pinf", "g30pinf"),
    ("graph hood, k=3", "graph", "gr3", "gr3"),
    ("graph hood, k=10", "graph", "gr10", "gr10"),
    ("offline Lloyd x3", "ref", "r64", "l64"),
    ("oracle", "ref", "r64", "oracle"),
]


def parse(path):
    out, build = {}, None
    if not path.exists():
        return out, build
    for line in path.read_text().splitlines():
        m = LINE.match(line)
        if m:
            out[(m.group(1), m.group(2))] = int(m.group(3))
        b = DONE.search(line)
        if b and line.startswith("DONE"):
            build = int(b.group(1))
    return out, build


def log_path(kind, tag, n):
    if kind == "ref":
        return REF / f"growth_D10_I0_{tag}_{n}_0.log"
    return ROOT / kind / f"every_{tag}_{n}_0.log"


def capped_costs():
    p = ROOT / "addcost.jsonl"
    if not p.exists():
        return {}
    return {
        r["arm"]: r["us_per_row"] for r in map(json.loads, p.read_text().splitlines())
    }


TIMED = {"r64": "base"}
OFFSETS = {"subtree pass": (-7, -2), "exact hood, k=30": (-6, 6)}


def collect():
    capped = capped_costs()
    rows = []
    for label, kind, tag, name in ARMS:
        ideal, walk, cost = {}, {}, None
        for n, nv in NS.items():
            d, build = parse(log_path(kind, tag, n))
            if (name, "ideal") in d:
                ideal[nv] = d[(name, "ideal")]
            if (name, "walk") in d:
                walk[nv] = d[(name, "walk")]
            if n == "1e6" and build is not None and (kind != "ref" or name == "r64"):
                cost = build * 1e6 / nv
        key = TIMED.get(tag, tag) if name == tag else None
        if key in capped:
            cost = capped[key]
        rows.append((label, ideal, walk, cost))
    return rows


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.grid(True, which="major", color="#e6e5e0", lw=0.6)


def panel_growth(ax, rows):
    for i, (label, ideal, _, _) in enumerate(rows):
        if not ideal:
            continue
        xs = sorted(ideal)
        ax.plot(
            xs, [ideal[x] for x in xs], "-o", color=SERIES[i], lw=2, ms=5, label=label
        )
    ax.plot([1e6], [38], marker="_", ms=22, mew=2, color=INK)
    ax.annotate("reject > 38", (1e6, 38), (1.6e6, 36), fontsize=6.5, color=INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N (rows)", fontsize=8, color=INK)
    ax.set_ylabel("leaves to last true neighbour (median)", fontsize=8, color=INK)
    ax.set_title("Partition quality (ideal centroid order)", fontsize=8.5, color=INK)
    style(ax)


def panel_cost(ax, rows):
    for i, (label, ideal, _, cost) in enumerate(rows):
        if cost is None or 1e6 not in ideal:
            continue
        ax.plot(cost, ideal[1e6], "o", color=SERIES[i], ms=7, mec="#fcfcfb", mew=1.5)
        short = (
            "tree-searched" if "tree-searched" in label else label.split(" every")[0]
        )
        dx, dy = OFFSETS.get(short, (6, 3))
        ax.annotate(
            short,
            (cost, ideal[1e6]),
            (dx, dy),
            textcoords="offset points",
            fontsize=6,
            color=INK,
            ha="right" if dx < 0 else "left",
        )
    ax.set_xlim(1, 3000)
    ax.axhline(38, color=INK, lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("add time per row at N=1e6 (us; 1 CPU scope)", fontsize=8, color=INK)
    ax.set_ylabel("leaves to last true neighbour at 1e6", fontsize=8, color=INK)
    ax.set_title("Quality bought per unit of add cost", fontsize=8.5, color=INK)
    style(ax)


def main():
    rows = collect()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.0),
        gridspec_kw={"width_ratios": [1, 1.05], "wspace": 0.75},
    )
    panel_growth(axes[0], rows)
    panel_cost(axes[1], rows)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=6.5, frameon=False)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.3, top=0.92, wspace=0.28)
    for ext in ("pdf", "png"):
        fig.savefig(OUT.with_suffix("." + ext), dpi=200)
    for label, ideal, walk, cost in rows:
        print(f"{label:34s} ideal {ideal} walk {walk} us/row {cost}")


main()
