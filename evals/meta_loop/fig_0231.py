import glob
import json
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "reports"
RUNS = [
    ("full scan, 4096 rows (0222)", f"{R}/report-0222-data/long_c4096_f16", "#999999"),
    ("run index (0223)", f"{R}/report-0223-data/long_c4096_f16", "#e69f00"),
    ("one sorted run (0225)", f"{R}/report-0225-data/long_c4096_f16", "#c0392b"),
    ("hybrid, 2 disk levels (0227)", f"{R}/report-0227-data/hyb_d2", "#2c7fb8"),
]


def load(d):
    out = []
    for f in glob.glob(d + "/*.jsonl"):
        with open(f) as fh:
            out += [json.loads(x) for x in fh]
    return sorted(out, key=lambda r: r["N"])


def smooth(N, y, bins=24):
    e = np.logspace(np.log10(N.min()), np.log10(N.max()), bins)
    ix = np.digitize(N, e)
    xs = [np.exp(np.log(N[ix == i]).mean()) for i in np.unique(ix)]
    ys = [y[ix == i].mean() for i in np.unique(ix)]
    return xs, ys


fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.7))
for nm, d, c in RUNS:
    r = load(d)
    if not r:
        continue
    N = np.array([x["N"] for x in r], float)
    for a, k in zip(ax[:2], ("query_us", "us_per_row")):
        a.plot(
            *smooth(N, np.array([x[k] for x in r])),
            "-o",
            ms=2.5,
            lw=1,
            color=c,
            label=nm,
        )
    if nm.startswith("one"):
        ax[2].plot(
            *smooth(N, np.array([x["query_us"] / x["leaf_only_us"] for x in r])),
            "-o",
            ms=2.5,
            color=c,
        )
for a, t in zip(
    ax,
    (
        "(a) query, lower is better",
        "(b) add, lower is better",
        "(c) query / one leaf read",
    ),
):
    a.set_xscale("log")
    a.set_title(t, fontsize=8)
    a.set_xlabel("N (rows)")
    a.axvline(1.5e6, color="k", ls=":", lw=0.8)
ax[0].set_yscale("log")
ax[1].set_yscale("log")
ax[0].set_ylabel("µs")
ax[0].legend(fontsize=5.5)
fig.tight_layout()
fig.savefig(sys.argv[1])
