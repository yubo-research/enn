import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("/mnt/scratch/enn_bench/res")
OUT = Path(sys.argv[1])
PROB = [
    (1000, "D = 1000 (target 1e6)"),
    (100, "D = 100 (target 1e7)"),
    (10, "D = 10 (target 1e8)"),
]
MET = [
    ("add_sync_us_row", "add time per row (µs)", 1.0),
    ("post_s", "posterior() time (s), 100 queries", 1.0),
    ("draw_s", "posterior_function_draw() time (s), 100 queries", 1.0),
]
BR = {"main": ("#d55e00", "o", "main"), "agg": ("#0072b2", "s", "dsweet/agg_obs")}


def load(v, d):
    p = R / f"{v}_D{d}.jsonl"
    rows = (
        [
            r
            for r in map(json.loads, p.read_text().splitlines())
            if "n" in r and "post_s" in r
        ]
        if p.exists()
        else []
    )
    if (v, d) == (
        "main",
        1000,
    ):
        first = [
            r
            for r in map(
                json.loads, (R / "main_D1000_first.jsonl").read_text().splitlines()
            )
            if "n" in r and "post_s" in r
        ]
        rows = first + [r for r in rows if r["n"] > max(x["n"] for x in first)]
    return rows


data = {(v, d): load(v, d) for v in BR for d, _ in PROB}
fig, ax = plt.subplots(3, 3, figsize=(10, 8.4), sharex="col")
for j, (d, title) in enumerate(PROB):
    for i, (k, lab, _) in enumerate(MET):
        a = ax[i, j]
        for v, (c, m, name) in BR.items():
            rows = data[(v, d)]
            rows = [r for r in rows if r.get(k) is not None]
            if rows:
                a.plot(
                    [r["n"] for r in rows],
                    [r[k] for r in rows],
                    c=c,
                    marker=m,
                    ms=4,
                    lw=1.2,
                    label=name,
                )
        a.set_xscale("log")
        a.set_yscale("log")
        a.grid(alpha=0.3, which="both", lw=0.3)
        if i == 0:
            a.set_title(title, fontsize=10)
        if j == 0:
            a.set_ylabel(lab, fontsize=8)
        if i == 2:
            a.set_xlabel("N (observations)")
        a.tick_params(labelsize=8)
ax[0, 0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "grid.pdf")
fig.savefig(OUT / "grid.png", dpi=110)
fig, ax = plt.subplots(3, 1, figsize=(3.4, 6.0), sharex=True)
names = ["add per row", "posterior()", "posterior_function_draw()"]
for i, (k, lab, _) in enumerate(MET):
    for (d, title), c in zip(PROB, ["#009e73", "#cc79a7", "#e69f00"]):
        a_, m_ = (
            {r["n"]: r[k] for r in data[("agg", d)] if r.get(k) is not None},
            {r["n"]: r[k] for r in data[("main", d)] if r.get(k) is not None},
        )
        ns = sorted(set(a_) & set(m_))
        if ns:
            ax[i].plot(
                ns,
                [m_[n] / a_[n] for n in ns],
                c=c,
                marker="o",
                ms=3,
                lw=1.1,
                label=f"D = {d}",
            )
    ax[i].axhline(1, c="k", lw=0.6)
    ax[i].set_xscale("log")
    ax[i].set_yscale("log")
    ax[i].grid(alpha=0.3, which="both", lw=0.3)
    ax[i].set_title(names[i], fontsize=9)
    ax[i].tick_params(labelsize=8)
    ax[i].set_ylabel("main / agg_obs", fontsize=8)
ax[2].set_xlabel("N (observations)")
ax[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "ratio.pdf")
fig.savefig(OUT / "ratio.png", dpi=110)
