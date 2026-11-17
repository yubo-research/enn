import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GREY, BLUE, GREEN, ORANGE, PURPLE = (
    "#9a9893",
    "#2a78d6",
    "#1baf7a",
    "#eb6834",
    "#6250d6",
)
RECALL_ARMS = [
    ("t3000", "tags R=3000 (bg10t)", GREY),
    ("r3000", "+ Lloyd retag", GREY),
    ("s3000", "+ re-parent", GREEN),
    ("sr3000", "+ re-parent + retag", GREEN),
    ("h3000", "+ re-home (0263)", ORANGE),
    ("hs3000", "+ re-home + re-parent", ORANGE),
    ("h300", "R=300 + re-home (h300)", PURPLE),
    ("hs300", "h300 + re-parent", PURPLE),
]
CAP_PANELS = [
    ("add_us_row", "add time (us / row)"),
    ("q_us_med", "posterior query, median (us)"),
    ("wr_b_row", "bytes written per added row"),
    ("q_rd_ops", "disk read requests per query"),
]
CAP_ARMS = [
    ("bg10t", "bg10t (tags R=3000)", GREY),
    ("s3000", "bg10t + re-parent", GREEN),
]


def load(p):
    return [json.loads(x) for x in open(p) if x.startswith("{")] if p.exists() else []


def style(ax):
    ax.grid(alpha=0.3, which="both", lw=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def rank_rows(d):
    g10 = load(d / "screen.jsonl")[-1]
    w16 = load(d / "screen_bg10t_w16.jsonl")[-1]
    live = load(d / "screen_bg10t_rp.jsonl")[-1]
    return [
        ("unbuffered g10", g10["base"], BLUE),
        ("bg10t as built", w16["base"], GREY),
        ("bg10t, offline\nleaf re-parent x3", w16["1:graph:16:3"], GREEN),
        ("bg10t, offline\nall levels x3", w16["9:graph:16:3"], GREEN),
        ("bg10t + live\nre-parent sweep", live["base"], ORANGE),
    ]


def fig_rank(d, out):
    rows = rank_rows(d)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    x = np.arange(len(rows))
    for i, (_, r, c) in enumerate(rows):
        a.bar(i - 0.18, r["owner_med"], 0.36, color=c)
        a.bar(i + 0.18, r["owner_p90"], 0.36, color=c, alpha=0.45)
        a.text(
            i - 0.18,
            r["owner_med"] + 1,
            f"{r['owner_med']:.0f}",
            ha="center",
            fontsize=6,
        )
        a.text(
            i + 0.18,
            r["owner_p90"] + 1,
            f"{r['owner_p90']:.0f}",
            ha="center",
            fontsize=6,
        )
        b.bar(i, r["parent_hit"], 0.6, color=c)
        b.text(
            i, r["parent_hit"] + 0.02, f"{r['parent_hit']:.2f}", ha="center", fontsize=6
        )
    for ax in (a, b):
        ax.set_xticks(x, [r[0] for r in rows], fontsize=5.5, rotation=30, ha="right")
        style(ax)
    a.axhline(3, color="k", lw=0.6, ls="--")
    a.set_ylabel("descent leaf's rank (0 = nearest)", fontsize=7)
    a.set_title("(a) owner rank: median (solid), p90 (light)", fontsize=8)
    b.set_ylabel("fraction of leaves", fontsize=7)
    b.set_title("(b) leaves under their nearest parent", fontsize=8)
    b.set_ylim(0, 1.1)
    fig.tight_layout()
    fig.savefig(out)


def fig_recall(d, out):
    by = {r["arm"]: r for r in load(d / "diag_1e6.jsonl")}
    rows = [(lab, by[a]["recall"], c) for a, lab, c in RECALL_ARMS if a in by]
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for i, (_, v, c) in enumerate(rows):
        ax.barh(i, v, color=c)
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=6)
    ax.axvline(0.99, color="k", lw=0.6, ls="--")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=6)
    ax.invert_yaxis()
    ax.set_xlim(0.9, 1.0)
    ax.set_xlabel("recall@10 at $10^6$ (300 queries, seed 0)", fontsize=7)
    style(ax)
    fig.tight_layout()
    fig.savefig(out)


def fig_cap(d, out):
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.2))
    for name, lab, c in CAP_ARMS:
        for rep, alpha, label in (
            ("_1e7", 1.0, lab),
            ("", 0.35, None),
            ("b", 0.35, None),
        ):
            recs = [
                r
                for r in load(d / "cap256" / f"capq_D10_256M_{name}{rep}.jsonl")
                if r["n"] >= 1000
            ]
            n = np.array([r["n"] for r in recs])
            for ax, (k, t) in zip(axs, CAP_PANELS):
                ax.plot(
                    n,
                    [r[k] if r[k] > 0 else np.nan for r in recs],
                    "o-",
                    ms=2.5,
                    lw=1,
                    color=c,
                    alpha=alpha,
                    label=label,
                )
    for ax, (_, t) in zip(axs, CAP_PANELS):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axvline(3.2e6, color="k", lw=0.5, ls=":")
        ax.set_title(t, fontsize=7)
        ax.set_xlabel("N", fontsize=7)
        ax.tick_params(labelsize=6)
        style(ax)
    axs[0].legend(fontsize=5.5, frameon=False)
    fig.tight_layout()
    fig.savefig(out)


def main():
    d = Path(sys.argv[1])
    fig_rank(d, d / "fig_0264a.pdf")
    fig_recall(d, d / "fig_0264b.pdf")
    if (d / "cap256" / "capq_D10_256M_s3000.jsonl").exists():
        fig_cap(d, d / "fig_0264c.pdf")


if __name__ == "__main__":
    main()
