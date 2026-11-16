import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#555555"
ARMS = [
    ("head0", "c30 ic10\n(old)", BLUE),
    ("B", "c30 ic30", BLUE),
    ("C", "c100 ic100\n(new)", ORANGE),
    ("pbuf", "c30 ic10\nbuffers", AQUA),
    ("D", "c100 ic100\nbuffers", AQUA),
]


def jsonl(p):
    p = Path(p)
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def by_seed(p, d=10, n=1_000_000):
    return {r["seed"]: r for r in jsonl(p) if r["D"] == d and r["n"] == n}


def paired(a, b, key):
    x = np.array([a[s][key] - b[s][key] for s in sorted(set(a) & set(b))])
    if len(x) < 2 or np.allclose(x, 0):
        return len(x), float(x.mean()) if len(x) else np.nan, 0.0, 1.0
    return (
        len(x),
        x.mean(),
        x.std(ddof=1) / np.sqrt(len(x)),
        stats.ttest_1samp(x, 0).pvalue,
    )


def quality_panel(ax, data, lines):
    for j, (arm, _, col) in enumerate(ARMS):
        n, m, se, p = paired(data[arm], data["main"], "loglik")
        ax.errorbar(j, m, yerr=se, fmt="o", color=col, ms=7, capsize=3, lw=2)
        ax.annotate(
            f"p={p:.2f}",
            (j, m + se),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=6,
        )
        for key in ("loglik", "nrmse"):
            n, m2, se2, p2 = paired(data[arm], data["main"], key)
            lines.append(
                f"{arm:6s} {key:7s} - main n={n:2d} {m2:+.5f} SE {se2:.5f} p {p2:.3f}"
            )
    ax.axhline(0, color=GRAY, lw=0.8, ls="--")
    ax.set_xticks(range(len(ARMS)), [a[1] for a in ARMS], fontsize=6)
    ax.set_ylabel("loglik − main (paired, 10 seeds)", fontsize=7)
    ax.set_title("(a) quality, D=10, N=1e6", fontsize=8)


def recall_panel(ax, qdir, data, lines):
    seed0 = jsonl(qdir / "seed0_sweep.jsonl")
    for tag, col, lab in (
        ("off", BLUE, "ic10, no buffers"),
        ("pbuf", AQUA, "ic10, buffers"),
    ):
        pts = sorted(
            (r["leaf_coef"], r["recall_at_10"]) for r in seed0 if r["arm"] == tag
        )
        ax.plot(*zip(*pts), "o-", color=col, lw=2, ms=6, label=f"seed 0, {lab}")
    for arm, c in (("B", 30), ("C", 100)):
        rec = [r["recall_at_10"] for r in data[arm].values()]
        lines.append(
            f"{arm} recall@10 mean {np.mean(rec):.4f} min {np.min(rec):.4f} n={len(rec)}"
        )
        ax.plot(
            c,
            np.mean(rec),
            "s",
            color=ORANGE,
            ms=8,
            label="ic=c, 10 seeds" if arm == "B" else None,
        )
    mr = next(r["recall_at_10"] for r in seed0 if r["arm"] == "main_recall")
    ax.axhline(mr, color=GRAY, ls="--", lw=0.8, label=f"main, c128 ({mr:.4f})")
    ax.set_xscale("log")
    ax.set_xlabel("leaf coefficient c (leaves = c·lnN)", fontsize=7)
    ax.set_ylabel("recall@10", fontsize=7)
    ax.set_title("(b) neighbour recall, D=10, N=1e6", fontsize=8)
    ax.legend(fontsize=6)


def timing_rows(tdir):
    t = {}
    for arm in ("c30", "c100", "main"):
        for r in jsonl(tdir / f"{arm}.jsonl"):
            t.setdefault((arm, r["D"], r["n"]), []).append(r)
    return t


def ram_line(arm, key, d, n, pairs):
    if key == "loglik":
        x = np.array([p[0][key] - p[1][key] for p in pairs])
        p = stats.ttest_1samp(x, 0).pvalue if not np.allclose(x, 0) else 1.0
        return f"RAM {arm}-main loglik D={d:3d} N={n:.0e} n={len(x)} {x.mean():+.5f} p {p:.3f}"
    r = np.array([p[0][key] / p[1][key] for p in pairs])
    g = np.exp(np.log(r).mean())
    return f"RAM {arm}/main {key:10s} D={d:3d} N={n:.0e} n={len(r)} ratio {g:.2f}"


def ram_ratios(t, lines):
    for arm in ("c30", "c100"):
        for key in ("query_s", "add_us_row", "loglik"):
            for a, d, n in sorted(k for k in t if k[0] == arm):
                pairs = list(zip(t[(a, d, n)], t[("main", d, n)]))
                lines.append(ram_line(arm, key, d, n, pairs))


def ram_panel(ax, t, lines):
    for arm, col in (("c30", BLUE), ("c100", ORANGE), ("main", GRAY)):
        for d, ls in ((10, "-"), (100, ":")):
            ns = sorted(n for (a, dd, n) in t if a == arm and dd == d)
            ys = [np.mean([r["query_s"] for r in t[(arm, d, n)]]) for n in ns]
            ax.plot(ns, ys, "o" + ls, color=col, lw=2, ms=5, label=f"{arm} D={d}")
    ram_ratios(t, lines)
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N", fontsize=7)
    ax.set_ylabel("1000 posterior queries, s", fontsize=7)
    ax.set_title("(c) in-RAM query time, 1 GB scope, 5 seeds", fontsize=8)
    ax.legend(fontsize=5.5, ncol=2)


def cold_panel(ax, cdir, lines):
    for arm, col in (("c30", BLUE), ("c100", ORANGE)):
        rows = jsonl(cdir / arm / "cold_D10_1G.jsonl")
        ns = [r["n"] for r in rows]
        q = [np.mean(r["cold_q_us"]) / 1e3 for r in rows]
        a = [np.mean(r["cold_add_us_row"]) / 1e3 for r in rows]
        ax.plot(ns, q, "o-", color=col, lw=2, ms=5, label=f"{arm} query")
        ax.plot(ns, a, "s:", color=col, lw=1.5, ms=4, label=f"{arm} add/row")
        for r, qi, ai in zip(rows, q, a):
            kib = np.mean(r["cold_q_rd_b"]) / 1024
            lines.append(
                f"cold {arm} N={r['n']:.0e} query {qi:.0f} ms read {kib:.0f} KiB add {ai:.2f} ms/row"
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N", fontsize=7)
    ax.set_ylabel("ms (store paged out before each op)", fontsize=7)
    ax.set_title("(d) cold cost, D=10, 1 GB, EBS, 1 run", fontsize=8)
    ax.legend(fontsize=6)


def main():
    qdir, sdir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    data = {a: by_seed(sdir / f"{a}.jsonl") for a in ("main", "head0", "pbuf")}
    data.update({a: by_seed(qdir / f"q_{a}.jsonl") for a in "BCD"})
    lines = []
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.6))
    quality_panel(ax[0, 0], data, lines)
    recall_panel(ax[0, 1], qdir, data, lines)
    ram_panel(ax[1, 0], timing_rows(qdir / "timing"), lines)
    cold_panel(ax[1, 1], qdir / "cold", lines)
    for a in ax.flat:
        a.tick_params(labelsize=6)
        a.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(out)
    out.with_suffix(".txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
