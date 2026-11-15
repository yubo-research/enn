import glob
import json
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, GREEN, GRAY, RED = "#1f6fb4", "#e07b1a", "#2b9348", "#888888", "#c0392b"


def load(f):
    with open(f) as fh:
        return [json.loads(l) for l in fh]


def fit(n, t, lo, hi):
    m = (n >= lo) & (n <= hi)
    A = np.vstack([np.ones(m.sum()), np.log(n[m])]).T
    coef, *_ = np.linalg.lstsq(A, t[m], rcond=None)
    r = t[m] - A @ coef
    s2 = r @ r / max(m.sum() - 2, 1)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(A.T @ A)))
    return coef, se


def panel_rep(ax, rep_dir):
    ns, ts = [], []
    for i, f in enumerate(sorted(glob.glob(f"{rep_dir}/bpa_D100_256M_r*.jsonl"))):
        g = [q for q in load(f) if "rows" in q and q["n"] >= 100]
        n = np.array([q["n"] for q in g], float)
        t = np.array([q["add_us_row"] for q in g])
        ax.plot(
            n,
            t,
            "o-",
            color=BLUE,
            alpha=0.35 + 0.2 * i,
            ms=3,
            lw=0.8,
            label="3 runs" if i == 0 else None,
        )
        ns.append(n)
        ts.append(t)
    n, t = np.concatenate(ns), np.concatenate(ts)
    (a, b), (sa, sb) = fit(n, t, 1e2, 1e6)
    xx = np.logspace(2, np.log10(3e6), 50)
    ax.plot(
        xx,
        a + b * np.log(xx),
        "--",
        color=ORANGE,
        lw=1.2,
        label="fit $a+b\\ln N$, $N\\leq10^6$",
    )
    pred = a + b * np.log(3e6)
    obs = t[n == 3e6]
    print(
        f"0234 fit 1e2..1e6: a={a:.1f}+-{sa:.1f} b={b:.2f}+-{sb:.2f}; pred(3e6)={pred:.1f}, obs={obs.mean():.0f}+-{obs.std(ddof=1) / np.sqrt(len(obs)):.0f}"
    )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="N",
        ylabel="add time (us/row)",
        title="(a) add, D=100, cap 256 MB, 3 runs",
    )
    ax.legend(fontsize=6)


def panel_cold(ax, data, key, scale, ylab, title, ab_key=None):
    ab = load(f"{data}/advise_ab.jsonl") if ab_key else []
    for D, c in ((10, BLUE), (100, GREEN)):
        fs = glob.glob(f"{data}/cold_D{D}_1G.jsonl")
        if not fs:
            continue
        g = load(fs[0])
        n = np.array([q["n"] for q in g], float)
        v = np.array([q[key] for q in g]) / scale
        med = np.median(v, 1)
        lo, hi = np.percentile(v, 25, 1), np.percentile(v, 75, 1)
        ax.errorbar(
            n,
            med,
            yerr=[med - lo, hi - med],
            fmt="o-",
            color=c,
            ms=3,
            lw=0.8,
            capsize=1.5,
            label=f"D={D} cold (median, IQR)",
        )
        if (n >= 1e4).sum() >= 3:
            (a, b), (sa, sb) = fit(n, med, 1e4, 1e7)
            xx = np.logspace(4, 9, 30)
            ax.plot(xx, a + b * np.log(xx), "--", color=c, lw=0.8, alpha=0.7)
            print(
                f"{key} D={D} fit N>=1e4: a={a:.2f}+-{sa:.2f} b={b:.2f}+-{sb:.2f} per lnN; pred 1e9 = {a + b * np.log(1e9):.1f}"
            )
        w = [q[ab_key] / scale for q in ab if q["D"] == D and q["advise"] == "normal"]
        if w:
            ax.plot(
                [1e5],
                [np.mean(w)],
                "*",
                color=c,
                ms=8,
                mec="k",
                mew=0.4,
                label=f"D={D} in RAM",
            )
    ax.set(xscale="log", yscale="log", xlabel="N", ylabel=ylab, title=title)
    ax.legend(fontsize=5.5)


def panel_ab(ax, data):
    r = load(f"{data}/advise_ab.jsonl")
    labs, vals, errs = [], [], []
    for D in (10, 100):
        for k, nm in (("add_us_row", "add"), ("q_us_med", "query")):
            a = np.array([q[k] for q in r if q["D"] == D and q["advise"] == "normal"])
            b = np.array([q[k] for q in r if q["D"] == D and q["advise"] == "random"])
            d = 100 * (b - a) / a
            labs.append(f"{nm}\nD={D}")
            vals.append(d.mean())
            errs.append(d.std(ddof=1) / np.sqrt(len(d)))
    ax.bar(range(4), vals, yerr=errs, color=[RED, GRAY, RED, GRAY], capsize=2)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(range(4), labs, fontsize=6)
    ax.set(
        ylabel="change in RAM (%, mean $\\pm$ SE)",
        title="(f) row MADV_RANDOM (rejected)",
    )


def panel_bytes(ax, data):
    fs = glob.glob(f"{data}/cold_D10_1G.jsonl")
    if not fs:
        return
    g = load(fs[0])
    n = [q["n"] for q in g]
    names = [
        "tree_codes.bin",
        "tree_meta.bin",
        "tree_slots.bin",
        "train_x.bin",
        "train_y.bin",
    ]
    cols = [BLUE, ORANGE, GREEN, GRAY, RED]
    bottom = np.zeros(len(g))
    x = np.arange(len(g))
    for nm, c in zip(names, cols):
        v = np.array([q["query_read_set_kib"].get(nm, 0) for q in g]) / 1024
        ax.bar(x, v, bottom=bottom, color=c, label=nm.replace(".bin", ""), width=0.8)
        bottom += v
    ax.set_xticks(x[::2], [f"{v:.0e}".replace("e+0", "e") for v in n][::2], fontsize=6)
    ax.set(
        xlabel="N",
        ylabel="MiB read by 1 cold query",
        title="(d) read set by file, D=10",
    )
    ax.legend(fontsize=6)


def panel_pf(ax, data):
    fs = glob.glob(f"{data}/prefetch_D10_1e5.jsonl")
    if not fs:
        return
    g = load(fs[0])
    keys = []
    for q in g:
        k = (q["code_prefetch"], q["prefetch_pages"], q["meta_prefetch"])
        if k not in keys:
            keys.append(k)
    for j, k in enumerate(keys):
        v = (
            np.concatenate(
                [
                    q["cold_q_us"]
                    for q in g
                    if (q["code_prefetch"], q["prefetch_pages"], q["meta_prefetch"])
                    == k
                ]
            )
            / 1e3
        )
        ax.bar(j, np.median(v), color=ORANGE if k == (10, 0, 4) else BLUE)
        ax.errorbar(
            j,
            np.median(v),
            yerr=1.2533 * v.std(ddof=1) / np.sqrt(len(v)),
            color="k",
            capsize=2,
        )
        print("prefetch", k, f"median {np.median(v):.0f} ms, n={len(v)}")
    ax.set_xticks(
        range(len(keys)), [f"{a}/{b}/{c}" for a, b, c in keys], rotation=45, fontsize=6
    )
    ax.set(
        ylabel="cold query (ms)",
        xlabel="code / pages / meta prefetch",
        title="(e) prefetch depth, D=10, N=1e5",
    )


def main():
    data, rep, out = sys.argv[1:4]
    _fig, ax = plt.subplots(2, 3, figsize=(10.5, 5.8))
    panel_rep(ax[0, 0], rep)
    panel_cold(
        ax[0, 1],
        data,
        "cold_q_us",
        1e3,
        "query time (ms)",
        "(b) one query, store evicted",
        "q_us_med",
    )
    ax[0, 1].set_ylim(1e-2, 2e3)
    panel_cold(
        ax[0, 2],
        data,
        "cold_add_us_row",
        1,
        "add time (us/row)",
        "(c) 32-row add, store evicted",
        "add_us_row",
    )
    panel_bytes(ax[1, 0], data)
    panel_pf(ax[1, 1], data)
    panel_ab(ax[1, 2], data)
    for a in ax.flat:
        a.grid(alpha=0.25, lw=0.4)
    plt.tight_layout()
    plt.savefig(out)


if __name__ == "__main__":
    main()
