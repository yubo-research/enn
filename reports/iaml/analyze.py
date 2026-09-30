"""Tables and figures for iaml.tex from results/stream.json and results/recall.json."""

import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ORDER = ["iso", "full", "B_doubling", "A_pool1", "A_pool3", "A_pool10", "C_online10"]
LABEL = {
    "iso": "Isotropic",
    "full": "Refit+rebuild every batch",
    "B_doubling": "B: doubling epochs",
    "A_pool1": "A: pool $K'=k$",
    "A_pool3": "A: pool $K'=3k$",
    "A_pool10": "A: pool $K'=10k$",
    "C_online10": "C: online, $K'=10k$",
}
PROBS = ["sparse", "friedman", "aniso", "iso"]


def mse(v):
    v = np.asarray(v)
    return v.mean(), v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0


def _metric_tables(g, nmax, prefix):
    for metric in ("test", "loo"):
        fh = open(f"{prefix}_{metric}.tex", "w")
        fh.write("\\begin{tabular}{l" + "r" * len(PROBS) + "}\\toprule\n")
        fh.write("Method & " + " & ".join(PROBS) + "\\\\\\midrule\n")
        for m in ORDER:
            cells = []
            for p in PROBS:
                mu, se = mse([r[metric] for r in g[(p, m, nmax)]])
                cells.append(f"${mu:.3f}\\pm{se:.3f}$")
            fh.write(LABEL[m] + " & " + " & ".join(cells) + "\\\\\n")
        fh.write("\\bottomrule\\end{tabular}\n")
        fh.close()


def _cost_table(rows, nmax, prefix):
    fh = open(f"{prefix}_cost.tex", "w")
    fh.write("\\begin{tabular}{lrr}\\toprule\nMethod & rows (re)indexed & fit+index seconds\\\\\\midrule\n")
    for m in ORDER:
        rs = [r for r in rows if r["method"] == m and r["n"] == nmax]
        fh.write(
            f"{LABEL[m]} & {np.mean([r['rows_indexed'] for r in rs]):,.0f} & {np.mean([r['seconds'] for r in rs]):.1f}\\\\\n"
        )
    fh.write("\\bottomrule\\end{tabular}\n")
    fh.close()


def stream_tables(rows, prefix):
    g = defaultdict(list)
    for r in rows:
        g[(r["problem"], r["method"], r["n"])].append(r)
    nmax = max(r["n"] for r in rows)
    _metric_tables(g, nmax, prefix)
    _cost_table(rows, nmax, prefix)
    _test_figure(rows, g, prefix)
    _weights_table(rows, nmax, prefix)


def _test_figure(rows, g, prefix):
    ns = sorted({r["n"] for r in rows})
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.2))
    for ax, p in zip(axs, PROBS):
        for m in ORDER:
            mu = [mse([r["test"] for r in g[(p, m, n)]])[0] for n in ns]
            ax.plot(ns, mu, marker="o", ms=3, label=LABEL[m].replace("$", "").replace("K'", "K'"))
        ax.set_xscale("log")
        ax.set_title(p)
        ax.set_xlabel("n (observations seen)")
    axs[0].set_ylabel("held-out log-lik / point")
    axs[-1].legend(fontsize=6, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{prefix}_fig.png", dpi=160)
    plt.close(fig)


def _weights_table(rows, nmax, prefix):
    fh = open(f"{prefix}_weights.tex", "w")
    rs = [r for r in rows if r["problem"] == "sparse" and r["n"] == nmax and r["seed"] == 0]
    fh.write("\\begin{tabular}{l" + "r" * 10 + "}\\toprule\nMethod & " + " & ".join(f"$x_{i}$" for i in range(10)))
    fh.write("\\\\\\midrule\n")
    for m in ORDER:
        r = [q for q in rs if q["method"] == m][0]
        fh.write(LABEL[m] + " & " + " & ".join(f"{v:.2g}" for v in r["a"]) + "\\\\\n")
    fh.write("\\bottomrule\\end{tabular}\n")
    fh.close()


def recall_table(rows, fh):
    fh.write("\\begin{tabular}{llrrrrrr}\\toprule\n")
    fh.write("Problem & Learner & R@$k$ & R@$3k$ & R@$10k$ & R@$30k$ & test (pool) & test (exact)\\\\\\midrule\n")
    for p in ("sparse", "friedman", "aniso"):
        for lrn in ("A_pool10", "C_online10"):
            rs = [r for r in rows if r["problem"] == p and r["learner"] == lrn]
            rec = [np.mean([r["recall"][str(mm)] for r in rs]) for mm in (1, 3, 10, 30)]
            tp, te = mse([r["test_pooled"] for r in rs]), mse([r["test_exact"] for r in rs])
            fh.write(
                f"{p} & {LABEL[lrn]} & "
                + " & ".join(f"{v:.2f}" for v in rec)
                + f" & ${tp[0]:.3f}\\pm{tp[1]:.3f}$ & ${te[0]:.3f}\\pm{te[1]:.3f}$\\\\\n"
            )
    fh.write("\\bottomrule\\end{tabular}\n")


if __name__ == "__main__":
    stream_tables(json.load(open("results/stream.json")), "t_v2")
    stream_tables(json.load(open("results/stream_v1_unsafe.json")), "t_v1")
    try:
        with open("t_recall.tex", "w") as fh:
            recall_table(json.load(open("results/recall.json")), fh)
    except FileNotFoundError:
        pass
