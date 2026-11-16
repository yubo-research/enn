import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

C = {"1000a": "#eb6834", "1000ax": "#2a78d6", "1000ap": "#1baf7a"}
LAB = {
    "1000a": "every level (0243 design)",
    "1000ax": "+ leaf extents",
    "1000ap": "+ buffer prefetch",
}
BANDS = [(1e6, 2e6), (2e6, 4e6), (4e6, 7e6), (7e6, 1.01e7)]


def rows(p):
    return (
        [json.loads(l) for l in Path(p).read_text().splitlines()]
        if Path(p).is_file()
        else []
    )


def checkpoints(r):
    return sorted(
        (o for o in r if "progress" not in o and o["n"] >= 1000), key=lambda o: o["n"]
    )


def progress(r):
    return sorted((o for o in r if "progress" in o), key=lambda o: o["n"])


def knee(r):
    c = [o for o in checkpoints(r) if 1e4 <= o["n"] <= 1e6]
    if len(c) < 3:
        return None
    x = np.log([o["n"] for o in c])
    b, a = np.polyfit(x, [o["add_us_row"] for o in c], 1)
    out = {"a": a, "b": b}
    for n in (3e6, 1e7):
        m = [o["add_us_row"] for o in checkpoints(r) if o["n"] == n]
        if m:
            out[f"meas_{n:.0e}"], out[f"pred_{n:.0e}"] = m[0], a + b * np.log(n)
    p = progress(r)
    for lo, hi in BANDS:
        v = [o["add_us_row"] for o in p if lo <= o["n"] < hi]
        rd = [o["rd_b_row"] for o in p if lo <= o["n"] < hi]
        if v:
            out[f"band_{lo:.0e}"] = (
                float(np.mean(v)),
                float(np.std(v) / np.mean(v)),
                float(np.mean(rd)),
            )
    return out


def faults(ax, d):
    r = sorted(
        rows(d / "stall_faults/stall_D100_256M_pb1000a.jsonl"), key=lambda o: o["n"]
    )
    if not r:
        return
    n = np.array([o["n"] for o in r], float)
    parts = [
        ("flt_take_row", "buffer take", "#eb6834"),
        ("flt_push_row", "child-buffer push", "#a35bd6"),
        ("flt_leaf_row", "leaf replay", "#2a78d6"),
    ]
    base = np.zeros(len(r))
    for k, lab, c in parts:
        v = np.array([o[k] for o in r])
        ax.fill_between(n, base, base + v, color=c, alpha=0.75, lw=0, label=lab)
        base += v
    ax.plot(n, [o["majflt_row"] for o in r], "k-", lw=0.8, label="all major faults")
    ax.set_xscale("log")
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_title("(c) major faults per add", fontsize=8, loc="left")
    ax.set_xlabel("N (rows)", fontsize=7)
    ax.set_ylabel("faults per added row", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(True, color="#e4e3df", lw=0.5)
    ax.axvline(1.2e6, color="#8c8c8c", lw=0.5, ls=":")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=5.5, frameon=False, loc="upper left")


def plot_run(ax, a, f, first):
    r = rows(f)
    if not r:
        return
    p, c = progress(r), checkpoints(r)
    lab = LAB[a] if first else None
    ax[0].plot(
        [o["n"] for o in p],
        [o["add_us_row"] for o in p],
        color=C[a],
        lw=0.8,
        alpha=0.8,
        label=lab,
    )
    ax[0].plot([o["n"] for o in c], [o["add_us_row"] for o in c], "o", color=C[a], ms=2)
    ax[1].plot(
        [o["n"] for o in p],
        [max(o["rd_b_row"], 1) for o in p],
        color=C[a],
        lw=0.8,
        alpha=0.8,
        label=lab,
    )
    k = knee(r)
    print(a, f.parent.name, json.dumps(k, default=float))
    if k and first:
        xs = np.logspace(4, 7, 30)
        ax[0].plot(xs, k["a"] + k["b"] * np.log(xs), "--", color=C[a], lw=0.7)


def style(x, t, y):
    x.set_xscale("log")
    x.set_yscale("log")
    x.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    x.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    x.set_title(t, fontsize=8, loc="left")
    x.set_xlabel("N (rows)", fontsize=7)
    x.set_ylabel(y, fontsize=7)
    x.tick_params(labelsize=6)
    x.grid(True, color="#e4e3df", lw=0.5)
    x.axvline(1.2e6, color="#8c8c8c", lw=0.5, ls=":")
    for s in ("top", "right"):
        x.spines[s].set_visible(False)
    x.legend(fontsize=6, frameon=False, loc="upper left")


def main():
    d = Path(sys.argv[1])
    runs = {a: sorted(d.glob(f"cap256/r*/bpa_D100_256M_pb{a}.jsonl")) for a in C}
    runs["1000a"].append(
        Path("reports/report-0243-data/cap256/bpa_D100_256M_pb1000a.jsonl")
    )
    fig, ax = plt.subplots(
        1, 3, figsize=(7.0, 2.5), gridspec_kw={"width_ratios": [1.15, 1, 1]}
    )
    for a, files in runs.items():
        for i, f in enumerate(files):
            plot_run(ax, a, f, i == 0)
    style(ax[0], "(a) add time, 256 MB cap", "us per added row")
    style(ax[1], "(b) disk reads during adds", "bytes read per added row")
    ax[1].set_xlim(8e5, 1.2e7)
    faults(ax[2], d)
    fig.tight_layout()
    fig.savefig(d / "fig_0244.pdf")


if __name__ == "__main__":
    main()
