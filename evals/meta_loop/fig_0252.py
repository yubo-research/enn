import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D_DIR, OUT = Path(sys.argv[1]), sys.argv[2]
C = {
    "base": "#4C72B0",
    "reassign": "#DD8452",
    "light": "#C44E52",
    "lloyd3": "#55A868",
    "kmeans": "#8172B3",
    "oracle": "#777777",
    "main": "#937860",
}


def rows(p):
    p = Path(p)
    return [json.loads(x) for x in open(p)] if p.exists() else []


def med(recs, key):
    return float(np.median([r[key] for r in recs]))


def screen_series(path, part, d=10, key="centroid"):
    pts = sorted(
        (o["n"], med(o["queries"], key), o.get("misassigned", np.nan))
        for o in rows(path)
        if o["part"] == part and o["D"] == d
    )
    return np.array(pts) if pts else np.zeros((0, 3))


def fig_a():
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    series = [
        ("tree (default)", D_DIR / "order_ll/order.jsonl", "tree", C["base"], "o-"),
        (
            "tree + reassign M=16",
            D_DIR / "order_r16/order.jsonl",
            "tree",
            C["reassign"],
            "s-",
        ),
        (
            "offline Lloyd x3",
            D_DIR / "order_ll/order.jsonl",
            "lloyd3",
            C["lloyd3"],
            "^--",
        ),
        ("offline 2-means", D_DIR / "order/order.jsonl", "kmeans", C["kmeans"], "v:"),
    ]
    for lab, p, part, col, sty in series:
        s = screen_series(p, part)
        if len(s):
            ax[0].plot(s[:, 0], s[:, 1], sty, color=col, label=lab, ms=4)
            if not np.isnan(s[:, 2]).all():
                ax[1].plot(s[:, 0], s[:, 2], sty, color=col, label=lab, ms=4)
    s = screen_series(D_DIR / "order_ll/order.jsonl", "tree", key="oracle")
    ax[0].plot(s[:, 0], s[:, 1], ":", color=C["oracle"], label="oracle order")
    ax[0].set(
        xscale="log",
        yscale="log",
        xlabel="N",
        ylabel="leaves to last true NN\n(centroid order, median)",
    )
    ax[1].set(
        xscale="log", xlabel="N", ylabel="rows not in nearest leaf", ylim=(0, 0.8)
    )
    ax[0].legend(fontsize=6.5, frameon=False)
    ax[0].set_title("D=10 uniform, 200 queries", fontsize=8)
    ax[1].set_title("misassigned fraction", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}a.pdf")
    fig.savefig(f"{OUT}a.png", dpi=150)


def trace_series(path, d, intr):
    out = []
    for o in rows(path):
        if o["D"] == d and o["intrinsic"] == intr:
            q = o["queries"]
            out.append(
                (
                    o["n"],
                    np.mean([r["leaves"] for r in q]),
                    np.median([r["last_true_rank"] for r in q]),
                    np.mean([r["recall_api"] for r in q]),
                )
            )
    return np.array(sorted(out)) if out else np.zeros((0, 4))


def plot_trace(a, s, sty, col, lab):
    if not len(s):
        return
    a.plot(s[:, 0], s[:, 1], sty, color=col, label=lab, ms=4)
    for n, l, _, rc in s:
        if rc < 0.9999:
            a.annotate(
                f"recall {rc:.4f}",
                (n, l),
                fontsize=6,
                color=col,
                xytext=(-40, 6),
                textcoords="offset points",
            )


def fig_b():
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    arms = [
        ("default", "trace_r0", C["base"], "o-"),
        ("reassign M=16 (light)", "trace_l16", C["light"], "s-"),
        ("reassign M=16 (full)", "trace_r16", C["reassign"], "d--"),
    ]
    for j, (d, intr, title) in enumerate(
        [(10, 0, "D=10 uniform"), (100, 5, "D=100, 5-dim data")]
    ):
        for lab, sub, col, sty in arms:
            s = trace_series(D_DIR / sub / "trace.jsonl", d, intr)
            plot_trace(ax[j], s, sty, col, lab)
        ax[j].set(xscale="log", yscale="log", xlabel="N", title=title)
    ax[0].set_ylabel("leaves visited per query (mean)")
    ax[0].legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}b.pdf")
    fig.savefig(f"{OUT}b.png", dpi=150)


def capq_series(path):
    pts = []
    for o in rows(path):
        if "n" in o and "q_us_mean" in o:
            pts.append(
                (
                    o["n"],
                    o.get("add_us_row", np.nan),
                    o["q_us_mean"] / 1e3,
                    o.get("q_rd_b", np.nan) / 1e6,
                )
            )
    return np.array(sorted(pts)) if pts else np.zeros((0, 4))


def fig_c():
    files = sorted((D_DIR / "cap256").glob("*.jsonl"))
    if not files:
        return
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for f in files:
        s = capq_series(f)
        if not len(s):
            continue
        lab = ("M=16 " if "r16" in f.stem else "default ") + f.stem[-1]
        col = C["light"] if "M=16" in lab else C["base"]
        sty = "s-" if "M=16" in lab else "o-"
        ax[0].plot(s[:, 0], s[:, 1], sty, color=col, label=lab, ms=3, alpha=0.85)
        ax[1].plot(s[:, 0], s[:, 2], sty, color=col, label=lab, ms=3, alpha=0.85)
    ax[0].set(xscale="log", yscale="log", xlabel="N", ylabel="add, us per row")
    ax[1].set(xscale="log", yscale="log", xlabel="N", ylabel="posterior, ms per query")
    ax[0].legend(fontsize=6, frameon=False)
    fig.suptitle("256 MB cgroup cap, 1 CPU, EBS, D=10 uniform", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}c.pdf")
    fig.savefig(f"{OUT}c.png", dpi=150)


fig_a()
fig_b()
fig_c()
