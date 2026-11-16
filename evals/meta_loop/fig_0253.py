import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA, OUT = Path(sys.argv[1]), sys.argv[2]
ARMS = {
    "base": ("default (one request per leaf)", "#4C72B0"),
    "blk": ("block read (one request per 16-page block)", "#DD8452"),
    "xblk": ("block read + leaf extents", "#55A868"),
}


def load(D):
    out = defaultdict(list)
    for f in sorted((DATA / "cap256").glob(f"capq_D{D}_256M_pb0_*.jsonl")):
        arm = f.stem.split("_pb0_")[1][:-1]
        out[arm].append([json.loads(line) for line in f.open()])
    return out


def panel(ax, runs, key, scale, ylabel, logy=True):
    for arm, (lab, col) in ARMS.items():
        for i, recs in enumerate(runs.get(arm, [])):
            r = [x for x in recs if x["n"] >= 1e3]
            ax.plot(
                [x["n"] for x in r],
                [x[key] * scale for x in r],
                "o-" if i == 0 else "s--",
                color=col,
                ms=3,
                lw=1.2,
                label=lab if i == 0 else None,
            )
    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("N (rows)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, which="both")


def cap_fig(D):
    runs = load(D)
    if not runs:
        return
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.2))
    panel(axs[0], runs, "q_us_mean", 1e-3, "posterior time (ms)")
    panel(axs[1], runs, "q_rd_ops", 1, "read requests / query")
    panel(axs[2], runs, "q_rd_b", 1e-6, "MB read / query")
    panel(axs[3], runs, "add_us_row", 1, "add time (us / row)")
    axs[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"256 MB cap, EBS, 1 CPU, D={D} uniform (solid: run a, dashed: run b)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}_cap_D{D}.{ext}", dpi=150)


def screen_fig():
    recs = [
        json.loads(f.read_text())["summary"]
        for f in sorted((DATA / "screen").glob("scr_x0_*.json"))
    ]
    if not recs:
        return
    recs.sort(key=lambda r: (r["D"], r["intrinsic"], r["N"]))
    lab = [f"D={r['D']}{' I5' if r['intrinsic'] else ''}\nN={r['N']:.0e}" for r in recs]
    x = np.arange(len(recs))
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
    w = 0.27
    for j, (key, name, col) in enumerate(
        [
            ("leaves", "leaves visited (= requests today)", "#4C72B0"),
            ("blocks", "distinct 16-page blocks", "#DD8452"),
            ("la10_runs", "runs within 10-leaf lookahead", "#8172B2"),
        ]
    ):
        axs[0].bar(x + (j - 1) * w, [r[key] for r in recs], w, label=name, color=col)
    axs[0].set_xticks(x, lab, fontsize=7)
    axs[0].set_ylabel("per query")
    axs[0].legend(fontsize=7)
    axs[1].bar(x, [1 / r["block_util"] for r in recs], 0.5, color="#C44E52")
    axs[1].set_xticks(x, lab, fontsize=7)
    axs[1].set_ylabel("bytes read: block / per-leaf")
    axs[1].axhline(1, color="k", lw=0.8)
    fig.suptitle("Offline walk screen (in RAM, seed 0, 100 queries, HEAD layout)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}_screen.{ext}", dpi=150)


COLS = {1e4: "#4C72B0", 1e5: "#DD8452", 3e5: "#8172B2", 1e6: "#55A868"}


def rows_at(recs, D, n):
    return sorted(
        [r for r in recs if r["D"] == D and r["N"] == n], key=lambda r: r["cap"]
    )


def logaxes(ax, xlabel, ylabel):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6)


def model_panel(ax, recs, D, I):
    for n in sorted({r["N"] for r in recs if r["D"] == D}):
        rr, c = rows_at(recs, D, n), COLS.get(float(n), "k")
        caps = [r["cap"] for r in rr]
        ax.plot(
            caps,
            [r["t_model_ms"] for r in rr],
            "o-",
            color=c,
            label=f"N={n:.0e} ideal order",
        )
        ax.plot(
            caps,
            [r["cert_t_model_ms"] for r in rr],
            "s--",
            color=c,
            label=f"N={n:.0e} exact stop",
        )
    ax.axvspan(30, 64, color="grey", alpha=0.15)
    ax.set_title(f"D={D}{' (5-dim subspace)' if I else ' uniform'}", fontsize=9)
    logaxes(ax, "leaf capacity (rows)", "modelled cold query (ms)")


def count_panel(ax, recs):
    for n in sorted({r["N"] for r in recs if r["D"] == 10}):
        rr, c = rows_at(recs, 10, n), COLS.get(float(n), "k")
        caps = [r["cap"] for r in rr]
        ax.plot(caps, [r["L"] for r in rr], "o-", color=c, label=f"N={n:.0e} requests")
        ax.plot(
            caps,
            [r["rows"] for r in rr],
            "^:",
            color=c,
            label=f"N={n:.0e} rows scanned",
        )
    logaxes(ax, "leaf capacity (rows)", "per query (ideal order), D=10")


def leafsize_fig():
    f = DATA / "leafsize" / "leafsize.jsonl"
    if not f.exists():
        return
    recs = [json.loads(line) for line in f.open()]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.3))
    model_panel(axs[0], recs, 10, 0)
    model_panel(axs[1], recs, 100, 5)
    count_panel(axs[2], recs)
    fig.suptitle(
        "Offline leaf-size screen: recursive 2-means leaves, EBS model max(ops/3000, bytes/125 MiB/s); grey = today's leaves"
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}_leafsize.{ext}", dpi=150)


screen_fig()
leafsize_fig()
for d in (10, 100):
    cap_fig(d)
