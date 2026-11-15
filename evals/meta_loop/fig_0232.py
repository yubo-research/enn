import json
import sys

import matplotlib
import numpy as np
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

d, out = sys.argv[1], sys.argv[2]
labels = {
    "main": "main 3246c4e",
    "off": "branch, no prefetch",
    "always": "branch, always prefetch",
    "gated": "branch, gated prefetch",
}


def load(k):
    with open(f"{d}/{k}.jsonl") as fh:
        return [json.loads(x) for x in fh]


R = {k: load(k) for k in labels}


def get(k, D, n, f):
    return np.array(
        [
            r[f]
            for r in sorted(R[k], key=lambda r: r["seed"])
            if r["D"] == D and r["n"] == n
        ]
    )


Ns = [1000, 10000, 100000]
fig, ax = plt.subplots(1, 3, figsize=(10, 3.2))
col = {"main": "#555555", "off": "#1f77b4", "always": "#7b3294", "gated": "#d95f02"}
rows = []
for j, D in enumerate((10, 100)):
    for k, lab in labels.items():
        m = [get(k, D, n, "add_us_row").mean() for n in Ns]
        se = [get(k, D, n, "add_us_row").std(ddof=1) / np.sqrt(5) for n in Ns]
        ax[j].errorbar(Ns, m, se, marker="o", color=col[k], label=lab, capsize=2)
    ax[j].set_xscale("log")
    ax[j].set_yscale("log")
    ax[j].set_title(f"add time, D={D}")
    ax[j].set_xlabel("N")
    ax[j].set_ylabel("us/row")
ax[0].legend(fontsize=7)
for n in Ns:
    for D in (10, 100):
        r = {"D": D, "n": n}
        for k in labels:
            r[k] = {
                f: get(k, D, n, f)
                for f in ("add_us_row", "query_s", "nrmse", "loglik", "maxrss_mb")
            }
        rows.append(r)
print(
    "D N | add us/row main/off/always/gated | query s main/gate | rss MB main/gate | dloglik gate-main (SE,p) | dnrmse (SE,p)"
)
dl = []
for r in rows:
    g, m, a, w = r["gated"], r["main"], r["off"], r["always"]

    def t(f, g=g, m=m):
        return g[f] - m[f]

    ll, nr = t("loglik"), t("nrmse")
    p = lambda x: stats.ttest_1samp(x, 0).pvalue if x.std() > 0 else 1.0
    se = lambda x: x.std(ddof=1) / np.sqrt(len(x))
    print(
        f"{r['D']} {r['n']:>6} | {m['add_us_row'].mean():.2f}/{a['add_us_row'].mean():.2f}/{w['add_us_row'].mean():.2f}/{g['add_us_row'].mean():.2f} | {m['query_s'].mean():.3f}/{g['query_s'].mean():.3f} | {m['maxrss_mb'].mean():.0f}/{g['maxrss_mb'].mean():.0f} | {ll.mean():+.4f} ({se(ll):.4f}, p={p(ll):.2f}) | {nr.mean():+.4f} ({se(nr):.4f}, p={p(nr):.2f})"
    )
    dl.append((r["D"], r["n"], g["query_s"].mean() / m["query_s"].mean()))
for D in (10, 100):
    ax[2].plot(Ns, [q for d_, n, q in dl if d_ == D], marker="o", label=f"D={D}")
ax[2].axhline(1, color="k", lw=0.5)
ax[2].set_xscale("log")
ax[2].set_title("query time, branch/main")
ax[2].set_xlabel("N")
ax[2].legend(fontsize=7)
fig.tight_layout()
fig.savefig(out)
