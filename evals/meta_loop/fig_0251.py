import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D, O = Path(sys.argv[1]), Path(sys.argv[2])
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, MUTED = "#222222", "#777777"
plt.rcParams.update(
    {
        "font.size": 7.5,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e6e6",
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.5,
        "legend.frameon": False,
    }
)


def read_jsonl(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def panel_last_true(ax, recs):
    d10 = sorted((r for r in recs if r["D"] == 10), key=lambda r: r["n"])
    for i, r in enumerate(d10):
        ltr = np.sort(
            [q["last_true_rank"] for q in r["queries"] if q["last_true_rank"] >= 0]
        )
        y = np.arange(1, len(ltr) + 1) / len(r["queries"])
        ax.step(ltr, y, where="post", color=C[i], label=f"N=1e{int(np.log10(r['n']))}")
        ax.axvline(r["budget"], color=C[i], ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("visit rank of leaf holding last true neighbour")
    ax.set_ylabel("fraction of queries")
    ax.set_title("(a) D=10: when the last true neighbour is found", loc="left", y=1.32)
    ax.legend(loc="lower right")


def panel_leaves(ax, recs):
    cases = [r for r in recs if not (r["D"] == 100 and r["intrinsic"] == 0)]
    lab = [
        f"D{r['D']}{'s5' if r['intrinsic'] else ''}\n1e{int(np.log10(r['n']))}"
        for r in cases
    ]
    keys = [
        ("leaves", "ball walk (visited)"),
        ("box_open", "exact box search"),
        ("last_true_rank", "last true nbr (p50)"),
        ("true_leaves", "leaves holding true k"),
    ]
    x = np.arange(len(cases))
    w = 0.2
    for j, (k, name) in enumerate(keys):
        v = [np.median([q[k] for q in r["queries"]]) for r in cases]
        ax.bar(x + (j - 1.5) * w, v, w * 0.9, color=C[j], label=name)
    ax.set_yscale("log")
    ax.set_xticks(x, lab)
    ax.set_ylabel("leaves per query (median)")
    ax.set_title("(b) leaves needed vs read", loc="left", y=1.32)
    ax.legend(fontsize=6, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.08))


def fig_a():
    recs = read_jsonl(D / "grid/trace.jsonl")
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
    panel_last_true(ax[0], recs)
    panel_leaves(ax[1], recs)
    fig.tight_layout()
    fig.savefig(O / "fig_0251a.pdf")
    fig.savefig(O / "fig_0251a.png", dpi=110)


def fig_b():
    st = json.loads((D / "qual/stats.json").read_text())
    cases = ["d10_1e5", "d10_1e6", "d100i5_1e5"]
    names = {
        "d10_1e5": "D=10, N=1e5",
        "d10_1e6": "D=10, N=1e6",
        "d100i5_1e5": "D=100 (5-dim), N=1e5",
    }
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for i, c in enumerate(cases):
        rows = sorted(
            (r for r in st if r["case"] == c and r["arm"] not in ("main",)),
            key=lambda r: r["leaves_walk"],
        )
        xs = [r.get("query_ratio_vs_p0", 1.0) for r in rows]
        dl = [r.get("dloglik_vs_main", [0, 0, 0, 1]) for r in rows]
        ax[0].errorbar(
            xs,
            [d[1] for d in dl],
            yerr=[d[2] for d in dl],
            color=C[i],
            marker="o",
            ms=4,
            capsize=2,
            label=names[c],
        )
        for r, xx, d in zip(rows, xs, dl):
            if r["arm"] == "0":
                continue
            ax[0].annotate(
                f"P={r['arm']}",
                (xx, d[1]),
                textcoords="offset points",
                xytext=(3, 3),
                fontsize=6,
                color=MUTED,
            )
        ax[1].plot(
            [r["leaves_walk"] for r in rows],
            [r["recall"] for r in rows],
            color=C[i],
            marker="o",
            ms=4,
            label=names[c],
        )
    ax[0].axhline(0, color=MUTED, lw=0.8)
    ax[0].set_xscale("log")
    ax[0].set_xticks([0.1, 0.2, 0.5, 1.0], ["0.1", "0.2", "0.5", "1"])
    ax[0].minorticks_off()
    ax[0].set_xlabel("posterior time / default (same build, in RAM)")
    ax[0].set_ylabel("loglik - main (paired, ±SE)")
    ax[0].set_title("(a) quality vs query cost; P=0 is the default", loc="left")
    ax[0].legend(fontsize=6, loc="lower right")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("leaves per walk")
    ax[1].set_ylabel("recall@10")
    ax[1].set_title("(b) recall vs leaves", loc="left")
    fig.tight_layout()
    fig.savefig(O / "fig_0251b.pdf")
    fig.savefig(O / "fig_0251b.png", dpi=110)


def fig_c():
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4))
    arms = [
        ("P0", "default P=0 (run 1)"),
        ("P0r", "default P=0 (run 2)"),
        ("P10r", "P=10"),
        ("P30", "P=30"),
    ]
    for i, (a, name) in enumerate(arms):
        f = D / f"cap256/capq_D100_256M_pb0_{a}.jsonl"
        if not f.exists():
            continue
        rs = read_jsonl(f)
        rs = [r for r in rs if r["n"] >= 100]
        n = [r["n"] for r in rs]
        ax[0].errorbar(
            n,
            [r["q_us_mean"] / 1e3 for r in rs],
            yerr=[r["q_us_se"] / 1e3 for r in rs],
            color=C[i],
            marker="o",
            ms=3,
            capsize=2,
            label=name,
        )
        nz = [r for r in rs if r["q_rd_b"] > 0]
        ax[1].plot(
            [r["n"] for r in nz],
            [r["q_rd_b"] / 1e3 for r in nz],
            color=C[i],
            marker="o",
            ms=3,
            label=name,
        )
    for a in ax:
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlabel("N (rows)")
    ax[0].set_ylabel("posterior ms / query")
    ax[1].set_ylabel("KB read from disk / query")
    ax[0].set_title("(a) queries under 256 MB cap, D=100 uniform", loc="left")
    ax[1].set_title("(b) disk reads per query (none below 3e4)", loc="left")
    ax[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(O / "fig_0251c.pdf")
    fig.savefig(O / "fig_0251c.png", dpi=110)


def main():
    fig_a()
    fig_b()
    fig_c()


if __name__ == "__main__":
    main()
