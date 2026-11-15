import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

COL = {"0": "#2a78d6", "1": "#eb6834", "2": "#1baf7a", "2n": "#e87ba4"}
LAB = {
    "0": "8-bit (default)",
    "1": "raw 4-bit",
    "2": "residual 4-bit",
    "2n": "residual, no refine-more",
}
MARK = {"0": "o", "1": "s", "2": "^", "2n": "v"}
INK, MUTED = "#222222", "#777777"


def load(d, a):
    out = {}
    for p in sorted(Path(d).glob(f"c4_D100_1G_a{a}_r*.jsonl")):
        r = p.stem.split("_r")[-1]
        for o in map(json.loads, p.read_text().splitlines()):
            out[(r, o["n"])] = o
    return out


def series(d, a, f):
    A = load(d, a)
    ns = sorted({k[1] for k in A})
    if f == "cold":
        v = [
            np.median(np.concatenate([A[k]["cold"] for k in A if k[1] == n])) / 1e3
            for n in ns
        ]
    elif f == "pages":
        v = [
            np.median(np.concatenate([A[k]["rd"] for k in A if k[1] == n])) / 4096
            for n in ns
        ]
    else:
        v = [
            np.median([A[k]["add_us_row"] or np.nan for k in A if k[1] == n])
            for n in ns
        ]
    return np.array(ns), np.array(v, float)


def ratio(d, a):
    A, B = load(d, a), load(d, "0")
    ns, m, se = [], [], []
    for n in sorted({k[1] for k in A} & {k[1] for k in B}):
        keys = [k for k in A if k[1] == n and k in B]
        lr = np.log(
            np.concatenate([A[k]["cold"] for k in keys])
            / np.concatenate([B[k]["cold"] for k in keys])
        )
        ns.append(n)
        m.append(np.exp(lr.mean()))
        se.append(np.exp(lr.mean()) * lr.std(ddof=1) / np.sqrt(len(lr)))
    return np.array(ns), np.array(m), np.array(se)


def style(ax, title, ylab, logy=True):
    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_title(title, fontsize=9, loc="left", color=INK)
    ax.set_xlabel("N", fontsize=8, color=MUTED)
    ax.set_ylabel(ylab, fontsize=8, color=MUTED)
    ax.tick_params(labelsize=7, colors=MUTED)
    ax.grid(True, which="major", color="#e6e6e6", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def lines(ax, d, arms, f):
    for a in arms:
        n, v = series(d, a, f)
        ax.plot(n, v, marker=MARK[a], ms=4, lw=1.5, color=COL[a], label=LAB[a])


def plain_ticks(ax, ticks):
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def ratios(ax, d, arms, title):
    for a in arms:
        if a == "0":
            continue
        n, m, se = ratio(d, a)
        ax.errorbar(
            n,
            m,
            yerr=se,
            marker=MARK[a],
            ms=4,
            lw=1.5,
            capsize=2,
            color=COL[a],
            label=LAB[a],
        )
    ax.axhline(1.0, color=MUTED, lw=0.8, ls="--")
    style(ax, title, "geo-mean ratio (±SE)")
    plain_ticks(ax, [0.5, 0.7, 1, 1.4, 2])
    ax.legend(fontsize=7, frameon=False)


def retries(ax, d):
    for a, rm, ls in (
        ("1", "0", "--"),
        ("1", "1", "-"),
        ("2", "0", "--"),
        ("2", "1", "-"),
    ):
        p = d / "diag" / f"m5_a{a}_rm{rm}.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(l) for l in p.read_text().splitlines()]
        ax.plot(
            [r["n"] for r in rows],
            [r["retries_per_q"] for r in rows],
            ls=ls,
            marker=MARK[a],
            ms=4,
            lw=1.5,
            color=COL[a],
            label=f"{LAB[a]}, refine-more {'on' if rm == '1' else 'off'}",
        )
    style(
        ax, "(d) f32 re-walks per query (warm), 5-dim", "re-walks / query", logy=False
    )
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    ax.set_ylim(-0.03, 1.3)


def main():
    d, out = Path(sys.argv[1]), sys.argv[2]
    m5, m0 = d / "v3_m5", d / "v3_m0"
    arms5 = [a for a in ("0", "1", "2", "2n") if load(m5, a)]
    arms0 = [a for a in ("0", "1", "2") if load(m0, a)]
    fig, axs = plt.subplots(2, 3, figsize=(10, 5.6))
    lines(axs[0, 0], m5, arms5, "cold")
    style(axs[0, 0], "(a) cold query, 5-dim data", "ms (median)")
    axs[0, 0].legend(fontsize=7, frameon=False)
    ratios(axs[0, 1], m5, arms5, "(b) cold ratio vs 8-bit, 5-dim")
    ratios(axs[1, 1], m0, arms0, "(e) cold ratio vs 8-bit, uniform")
    lines(axs[0, 2], m5, arms5, "pages")
    style(axs[0, 2], "(c) pages read per cold query, 5-dim", "4 KiB pages (median)")
    retries(axs[1, 0], d)
    lines(axs[1, 2], m5, arms5, "add")
    style(axs[1, 2], "(f) add time, 5-dim (1 GB cap)", "us/row (median)")
    plain_ticks(axs[1, 2], [10, 30, 100, 300])
    fig.tight_layout()
    fig.savefig(out)
    fig.savefig(str(out).rsplit(".", 1)[0] + ".png", dpi=80)


if __name__ == "__main__":
    main()
