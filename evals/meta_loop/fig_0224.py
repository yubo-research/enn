import glob
import json
import pathlib
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(f):
    r = [json.loads(x) for x in pathlib.Path(f).read_text().splitlines()]
    return np.array([x["N"] for x in r], float), r


d22, d23, d24, rf, out = sys.argv[1:6]
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.6))
S = {
    "scan (0222)": (d22, "--"),
    "run index (0223)": (d23, ":"),
    "one sorted run (this)": (d24, "-"),
}
for c, col in ((1024, "C0"), (4096, "C1")):
    for nm, (d, ls) in S.items():
        fs = glob.glob(f"{d}/long_c{c}_f16/*.jsonl")
        if not fs:
            continue
        N, r = load(fs[0])
        for a, k in zip(ax[:2], ("us_per_row", "query_us")):
            t = np.convolve([x[k] for x in r], np.ones(5) / 5, "valid")
            a.plot(
                N[2:-2],
                t,
                color=col,
                ls=ls,
                label=f"{c} rows, {nm}" if k == "query_us" else None,
            )
for a, y in zip(ax[:2], ("add time per row (us)", "query time (us)")):
    a.set(xscale="log", yscale="log", xlabel="N (rows)", ylabel=y)
    a.grid(alpha=0.3)
ax[1].legend(fontsize=5)
g = [json.loads(x) for x in pathlib.Path(rf).read_text().splitlines()]
ax[2].plot([x["gb"] for x in g], [x["pread_us"] for x in g], "ko-")
ax[2].axvline(0.25, color="gray", ls=":")
ax[2].set_xticks([x["gb"] for x in g])
ax[2].set_xticklabels([f"{x['gb']:g}" for x in g])
ax[2].minorticks_off()
ax[2].set(
    xscale="log",
    yscale="log",
    xlabel="file size (GB), cap 256 MB",
    ylabel="8 KB random pread (us)",
)
ax[2].grid(alpha=0.3)
for a, s in zip(ax, "abc"):
    a.set_title(f"({s})", fontsize=8, loc="left")
fig.tight_layout()
fig.savefig(out)
