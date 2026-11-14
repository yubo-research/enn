import json
import math
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

D = sys.argv[1]
ORANGE, BLUE, GREEN, GREY = "#E69F00", "#0072B2", "#009E73", "#777777"
plt.rcParams.update(
    {"font.size": 8, "axes.spines.top": False, "axes.spines.right": False}
)
b = [json.loads(l) for l in Path(D, "recall.jsonl").read_text().splitlines()]
m = [json.loads(l) for l in Path(D, "main", "quality.jsonl").read_text().splitlines()]


def ms(v, f):
    x = np.array([r[f] for r in v])
    se = x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else 0
    return x.mean(), se


fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4))
leaves = [3, 10, 30, 100]
for Dd, c in [(10, BLUE), (100, ORANGE)]:
    rec, ll, llse = [], [], []
    for L in leaves:
        v = [r for r in b if r["D"] == Dd and r["n"] == 100000 and r["leaf_coef"] == L]
        rec.append(ms(v, "recall_at_10")[0])
        mu, se = ms(v, "loglik")
        ll.append(mu)
        llse.append(se)
    ax[0].plot(leaves, rec, "o-", color=c, label=f"D={Dd}")
    ax[1].errorbar(leaves, ll, yerr=llse, fmt="o-", color=c, capsize=2, label=f"D={Dd}")
    if Dd == 10:
        ax[1].axhline(
            ms([r for r in m if r["D"] == 10 and r["n"] == 100000], "loglik")[0],
            color=GREY,
            ls="--",
            label="main, D=10",
        )
    else:
        ax[1].axhline(
            ms([r for r in m if r["D"] == 100 and r["n"] == 100000], "loglik")[0],
            color=GREY,
            ls=":",
            label="main, D=100",
        )
for a in ax[:2]:
    a.set_xscale("log")
    a.set_xticks(leaves)
    a.set_xticklabels(leaves)
    a.set_xlabel("leaf coefficient (leaves = c ln N)")
ax[0].set_ylabel("recall@10 vs exact")
ax[0].set_ylim(0, 1.02)
ax[0].legend(frameon=False)
ax[1].set_ylabel("loglik (mean ± SE)")
ax[1].legend(frameon=False, fontsize=6)
mk = {(r["D"], r["n"], r["seed"]): r for r in m}
bd = {(r["D"], r["n"], r["seed"]): r for r in b if r["leaf_coef"] == 10.0}
for Dd, c in [(10, BLUE), (100, ORANGE)]:
    xs, ys, es = [], [], []
    for N in [1000, 10000, 100000]:
        d = [
            bd[k]["loglik"] - mk[k]["loglik"]
            for k in bd
            if k[0] == Dd and k[1] == N and k in mk
        ]
        xs.append(N)
        ys.append(np.mean(d))
        es.append(np.std(d, ddof=1) / math.sqrt(len(d)))
    ax[2].errorbar(xs, ys, yerr=es, fmt="o-", color=c, capsize=2, label=f"D={Dd}")
ax[2].axhline(0, color=GREY, lw=0.8)
ax[2].set_xscale("log")
ax[2].set_xlabel("N")
ax[2].set_ylabel("loglik(branch) − loglik(main)")
ax[2].legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(D, "fig_recall.pdf"))

fig, ax = plt.subplots(figsize=(3.4, 2.4))
for Dd, c in [(10, BLUE), (100, ORANGE)]:
    t = [
        ms(
            [r for r in b if r["D"] == Dd and r["n"] == 100000 and r["leaf_coef"] == L],
            "neighbors_ms_per_query",
        )[0]
        for L in leaves
    ]
    ax.plot(leaves, t, "o-", color=c, label=f"D={Dd}")
ax.set_xscale("log")
ax.set_xticks(leaves)
ax.set_xticklabels(leaves)
ax.set_yscale("log")
ax.set_xlabel("leaf coefficient c (N=1e5)")
ax.set_ylabel("neighbour search, ms/query")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(D, "fig_cost.pdf"))

out = []
for Dd in [10, 100]:
    for N in [1000, 10000, 100000]:
        keys = [k for k in bd if k[0] == Dd and k[1] == N and k in mk]
        for met in ["loglik", "nrmse"]:
            d = np.array([bd[k][met] - mk[k][met] for k in keys])
            if d.std() == 0:
                out.append((Dd, N, met, "default", 0, 0, 1))
                continue
            p = stats.ttest_1samp(d, 0).pvalue
            out.append(
                (Dd, N, met, "default", d.mean(), d.std(ddof=1) / math.sqrt(len(d)), p)
            )
for Dd in [10, 100]:
    for L in [10, 30, 100]:
        keys = [
            k
            for k in mk
            if k[0] == Dd
            and k[1] == 100000
            and any(
                r["seed"] == k[2]
                and r["leaf_coef"] == L
                and r["D"] == Dd
                and r["n"] == 100000
                for r in b
            )
        ]
        br = {
            r["seed"]: r
            for r in b
            if r["D"] == Dd and r["n"] == 100000 and r["leaf_coef"] == L
        }
        d = np.array([br[s]["loglik"] - mk[(Dd, 100000, s)]["loglik"] for s in br])
        p = stats.ttest_1samp(d, 0).pvalue
        out.append(
            (
                Dd,
                100000,
                "loglik",
                f"leaf{L}",
                d.mean(),
                d.std(ddof=1) / math.sqrt(len(d)),
                p,
            )
        )
for o in out:
    print("{} {} {} {} diff={:.4f} se={:.4f} p={:.3f}".format(*o))
