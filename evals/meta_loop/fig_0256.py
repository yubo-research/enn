import glob
import json
import math
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR, OUT = sys.argv[1], sys.argv[2]
THR = [float(t) for t in os.environ.get("THR", "0.01,0.003").split(",")]
REC = [float(t) for t in os.environ.get("REC", "0.99,0.999").split(",")]
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
GRAY = "#52514e"
GRID_COLOR = "#e6e6e3"
NAMES = {
    ("r64", "walk"): "tree, real walk (64-row leaves)",
    ("r64r", "walk"): "tree + split-time reassignment, real walk",
    ("r256", "walk"): "tree, 256-row leaves, real walk",
    ("r64", "ideal"): "tree partition, ideal centroid order",
    ("o64", "ideal"): "offline 2-means, 64 rows, ideal order",
    ("l64", "ideal"): "2-means + 3 Lloyd passes, ideal order",
    ("o256", "ideal"): "offline 2-means, 256 rows, ideal order",
    ("o1024", "ideal"): "offline 2-means, 1024 rows, ideal order",
    ("oracle", "ideal"): "oracle: leaves holding the true k",
}
WALK = [("r64", "walk"), ("r64r", "walk"), ("r256", "walk")]
IDEAL = [
    ("r64", "ideal"),
    ("o64", "ideal"),
    ("l64", "ideal"),
    ("o256", "ideal"),
    ("o1024", "ideal"),
]
ORACLE = ("oracle", "ideal")
ALL = WALK + IDEAL + [ORACLE]


def load():
    rows = []
    for d in DIR.split(":"):
        for f in glob.glob(os.path.join(d, "growth_*.jsonl")):
            with open(f) as fh:
                rows += [json.loads(x) for x in fh]
    return [r for r in rows if r["arm"] != "oracle" or "r64" in r["rows_per_leaf"]]


def by_key(rows):
    out = {}
    for r in rows:
        out.setdefault((r["arm"], r["order"]), {}).setdefault(r["N"], []).append(r)
    return out


def pooled(recs):
    dm = np.mean([r["dmse_grid"] for r in recs], axis=0)
    se = np.sqrt(np.sum([np.square(r["dmse_se_grid"]) for r in recs], axis=0))
    return {
        "g": np.asarray(recs[0]["grid"]),
        "dm": dm,
        "se": se / len(recs),
        "rc": np.mean([r["recall_grid"] for r in recs], axis=0),
        "ex": float(np.mean([r["ex_mse"] for r in recs])),
    }


def ppl_one(r, arm):
    a = arm if arm in r["rows_per_leaf"] else next(iter(r["rows_per_leaf"]))
    if "pages_per_leaf" in r and a in r["pages_per_leaf"]:
        return float(r["pages_per_leaf"][a])
    return float(math.ceil((64 + r["rows_per_leaf"][a] * (8 + r["D"])) / 4096))


def leaves_at(p, thr, by_recall):
    ok = np.flatnonzero(p["rc"] >= thr if by_recall else p["dm"] <= thr * p["ex"])
    return int(p["g"][ok[0]]) if len(ok) else -1


def power_fit(ns, ys):
    p = np.polyfit(np.log(ns), np.log(ys), 1)
    pred = np.exp(np.polyval(p, np.log(ns)))
    return float(p[0]), float(np.sqrt(np.mean((np.log(ys) - np.log(pred)) ** 2)))


def ln_fit(ns, ys):
    q = np.polyfit(np.log(ns), ys, 1)
    pred = np.maximum(np.polyval(q, np.log(ns)), 1e-9)
    rms = float(np.sqrt(np.mean((np.log(ys) - np.log(pred)) ** 2)))
    return float(q[0]), float(q[1]), rms


def fits(ns, ys):
    ns, ys = np.asarray(ns, float), np.asarray(ys, float)
    ok = (ys > 0) & (ns > 0)
    ns, ys = ns[ok], ys[ok]
    keys = ("exp", "rms_pow", "b_ln", "a_ln", "rms_ln", "local_exp_last")
    if len(ns) < 3:
        return dict.fromkeys(keys, float("nan"))
    e, rp = power_fit(ns, ys)
    b, a, rl = ln_fit(ns, ys)
    loc = float(np.log(ys[-1] / ys[-2]) / np.log(ns[-1] / ns[-2]))
    return dict(zip(keys, (e, rp, b, a, rl, loc)))


def series(bk, key, field):
    d = bk.get(key, {})
    ns = sorted(d)
    vals = [float(np.mean([r[field] for r in d[n]])) for n in ns]
    lo = [float(np.min([r[field] for r in d[n]])) for n in ns]
    hi = [float(np.max([r[field] for r in d[n]])) for n in ns]
    return ns, vals, lo, hi


def parity_series(bk, key, thr, by_recall=False):
    d = bk.get(key, {})
    ns, leaves, pages = [], [], []
    for n in sorted(d):
        m = leaves_at(pooled(d[n]), thr, by_recall)
        ppl = float(np.mean([ppl_one(r, key[0]) for r in d[n]]))
        ns.append(n)
        leaves.append(m)
        pages.append(float(m * ppl) if m > 0 else -1)
    return ns, leaves, pages


def style_axis(ax, title, xlabel, ylabel):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9.5, loc="left")
    ax.grid(True, which="both", color=GRID_COLOR, lw=0.6)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")


def panel(ax, bk, keys, title, ylabel):
    for i, key in enumerate(keys):
        ns, v, lo, hi = series(bk, key, "ltr_p50")
        if ns:
            ax.plot(ns, v, "-o", color=PAL[i % 6], ms=4, lw=1.6, label=NAMES[key])
            ax.fill_between(ns, lo, hi, color=PAL[i % 6], alpha=0.15, lw=0)
    ns, v, lo, hi = series(bk, ORACLE, "ltr_p50")
    if ns:
        ax.plot(ns, v, "--", color=GRAY, lw=1.4, label=NAMES[ORACLE])
    style_axis(ax, title, "N (rows)", ylabel)


def fig_rank(bk):
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
    ylab = "median leaves to last true neighbour (lower = better)"
    panel(axs[0], bk, WALK, "(a) Real walk order", ylab)
    panel(axs[1], bk, IDEAL, "(b) Ideal centroid order on each partition", "")
    fig.tight_layout()
    fig.savefig(f"{OUT}_rank.pdf")
    fig.savefig(f"{OUT}_rank.png", dpi=150)
    plt.close(fig)


def parity_panel(ax, bk, thr):
    for i, key in enumerate(WALK + IDEAL[1:]):
        ns, lv, pg = parity_series(bk, key, thr, True)
        pts = [(n, p) for n, m, p in zip(ns, lv, pg) if m > 0]
        if pts:
            mk = "-o" if key[1] == "walk" else "--s"
            col = PAL[i % 6] if i < 6 else GRAY
            ax.plot(*zip(*pts), mk, color=col, ms=4, lw=1.6, label=NAMES[key])
    ns, lv, pg = parity_series(bk, ORACLE, thr, True)
    if ns:
        ax.plot(ns, pg, ":", color=GRAY, lw=1.4, label=NAMES[ORACLE])
    title = f"Pages read until pooled recall@10 >= {thr:g} (lower = better)"
    style_axis(ax, title, "N (rows)", "4 KiB pages per query")


def fig_parity(bk):
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.9))
    for ax, thr in zip(axs, REC[:2]):
        parity_panel(ax, bk, thr)
    fig.tight_layout()
    fig.savefig(f"{OUT}_parity.pdf")
    fig.savefig(f"{OUT}_parity.png", dpi=150)
    plt.close(fig)


def curve_lines(axs, n, p, c):
    pct = 100 / p["ex"]
    axs[0].plot(p["g"], p["rc"], "-", color=c, lw=1.6, label=f"N = {n:.0e}")
    axs[1].plot(p["g"], pct * p["dm"], "-", color=c, lw=1.6, label=f"N = {n:.0e}")
    lo, hi = pct * (p["dm"] - p["se"]), pct * (p["dm"] + p["se"])
    axs[1].fill_between(p["g"], lo, hi, color=c, alpha=0.15, lw=0)


def fig_curves(bk):
    d = bk.get(("r64", "walk"), {})
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.7))
    for i, n in enumerate(sorted(d)):
        curve_lines(axs, n, pooled(d[n]), PAL[i % 6])
    axs[0].set_ylabel("recall@10 after m leaves")
    axs[1].set_ylabel("excess MSE of the k-NN prediction, % of exact")
    axs[1].axhline(1.0, color=GRAY, lw=0.8, ls="--")
    axs[1].set_ylim(-1, 15)
    titles = [
        "(a) Recall vs leaves visited, real walk (higher = better)",
        "(b) Prediction excess error vs leaves visited (lower = better)",
    ]
    for ax, t in zip(axs, titles):
        ax.set_xscale("log")
        ax.set_xlabel("leaves visited, m (walk order)")
        ax.set_title(t, fontsize=9.5, loc="left")
        ax.grid(True, which="both", color=GRID_COLOR, lw=0.6)
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}_curves.pdf")
    fig.savefig(f"{OUT}_curves.png", dpi=150)
    plt.close(fig)


def fmt_fit(ft):
    return (
        f"{ft['exp']:.3f} | {ft['rms_pow']:.3f} | {ft['b_ln']:.1f} | "
        f"{ft['rms_ln']:.3f} | {ft['local_exp_last']:.2f}"
    )


def rank_table(bk):
    lines = [
        "arm | order | N: median ltr [min-max over seeds](seeds) | pow exp | rms_pow | b per lnN | rms_ln | local exp last two N"
    ]
    for key in ALL:
        ns, v, lo, hi = series(bk, key, "ltr_p50")
        if ns:
            pts = " ".join(
                f"{n:.0e}:{x:.0f}[{a:.0f}-{b:.0f}]({len(bk[key][n])})"
                for n, x, a, b in zip(ns, v, lo, hi)
            )
            lines.append(f"{key[0]} | {key[1]} | {pts} | {fmt_fit(fits(ns, v))}")
    return lines


def parity_table(bk, thr, byr):
    head = (
        f"leaves / pages until recall >= {thr:g}"
        if byr
        else f"parity leaves / pages at dMSE <= {100 * thr:g}% of exact MSE"
    )
    lines = [
        "",
        head
        + ": arm | order | N:leaves/pages ... | pow exp | rms_pow | b per lnN | rms_ln | local exp",
    ]
    for key in ALL:
        ns, lv, pg = parity_series(bk, key, thr, byr)
        if ns:
            ft = fits([n for n, m in zip(ns, lv) if m > 0], [m for m in lv if m > 0])
            pts = " ".join(f"{n:.0e}:{m}/{p:.0f}" for n, m, p in zip(ns, lv, pg))
            lines.append(f"{key[0]} | {key[1]} | {pts} | {fmt_fit(ft)}")
    return lines


def run_table(bk):
    lines = [
        "",
        "r64 walk per N: budget leaves L, ex_mse, miss, n_leaves, build_s, total_s",
    ]
    fields = ("walk_leaves_mean", "ex_mse", "ltr_miss", "build_s", "total_s")
    for n, recs in sorted(bk.get(("r64", "walk"), {}).items()):
        m = {k: float(np.mean([r[k] for r in recs])) for k in fields}
        lines.append(
            f"{n:.0e}: L {m['walk_leaves_mean']:.0f} ex_mse {m['ex_mse']:.4f} miss {m['ltr_miss']:.3f} "
            f"n_leaves {recs[0]['n_leaves']} build_s {m['build_s']:.0f} total_s {m['total_s']:.0f}"
        )
    return lines


def n_label(n):
    e = math.floor(math.log10(n) + 1e-9)
    m = n / 10**e
    return f"$10^{{{e}}}$" if abs(m - 1) < 1e-6 else f"${m:.0f}\\times10^{{{e}}}$"


def tex_row(label, ns, vals, ft, all_ns):
    cells = {n: v for n, v in zip(ns, vals)}
    body = " & ".join(
        f"{cells[n]:.0f}" if n in cells and cells[n] > 0 else "--" for n in all_ns
    )
    return f"{label} & {body} & {ft['exp']:.2f} & {ft['b_ln']:.1f} \\\\"


def latex_tables(bk):
    all_ns = sorted({n for key in ALL for n in bk.get(key, {})})
    ncol = len(all_ns) + 3
    head = " & ".join(n_label(n) for n in all_ns)
    out = [
        f"\\begin{{tabular}}{{l{'r' * (ncol - 1)}}}\\toprule",
        f"order & {head} & exp & $b$ \\\\\\midrule",
    ]
    out.append(
        f"\\multicolumn{{{ncol}}}{{l}}{{\\emph{{median leaves to the last true neighbour}}}}\\\\"
    )
    for key in ALL:
        ns, v, _lo, _hi = series(bk, key, "ltr_p50")
        if ns:
            out.append(tex_row(NAMES[key], ns, v, fits(ns, v), all_ns))
    for thr in REC:
        out.append(
            f"\\midrule\\multicolumn{{{ncol}}}{{l}}{{\\emph{{smallest budget with pooled recall@10 $\\ge$ {thr:g}}}}}\\\\"
        )
        for key in ALL:
            ns, lv, _pg = parity_series(bk, key, thr, True)
            if ns:
                ft = fits(
                    [n for n, m in zip(ns, lv) if m > 0], [m for m in lv if m > 0]
                )
                out.append(tex_row(NAMES[key], ns, lv, ft, all_ns))
    out.append("\\bottomrule\\end{tabular}")
    with open(f"{OUT}_table.tex", "w") as fh:
        fh.write("\n".join(out) + "\n")


def main():
    bk = by_key(load())
    fig_rank(bk)
    fig_parity(bk)
    fig_curves(bk)
    lines = rank_table(bk)
    for thr in REC:
        lines += parity_table(bk, thr, True)
    for thr in THR:
        lines += parity_table(bk, thr, False)
    lines += run_table(bk)
    latex_tables(bk)
    text = "\n".join(lines)
    with open(f"{OUT}_summary.txt", "w") as fh:
        fh.write(text + "\n")
    print(text)


main()
