import glob
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

D, OUT = sys.argv[1], sys.argv[2]


def load(p):
    return [json.loads(x) for x in pathlib.Path(p).read_text().splitlines()]


fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
for i, f in enumerate(sorted(glob.glob(f"{D}/rep*/bs_buffered_*.jsonl"))):
    r = load(f)
    ax[0].plot(
        [x["N"] for x in r],
        [x["us_per_row"] for x in r],
        color="#1f77b4",
        lw=1,
        alpha=0.8,
        label="buffered (2 runs)" if i == 0 else None,
    )
    ax[1].plot(
        [x["N"] for x in r],
        [x["write_B_row"] for x in r],
        color="#1f77b4",
        lw=1,
        alpha=0.8,
    )
    ax[1].plot(
        [x["N"] for x in r],
        [x["read_B_row"] for x in r],
        color="#1f77b4",
        lw=1,
        ls=":",
        alpha=0.8,
    )
for f in glob.glob(f"{D}/first/bs_inplace_*.jsonl"):
    r = load(f)
    ax[0].plot(
        [x["N"] for x in r],
        [x["us_per_row"] for x in r],
        color="#d62728",
        lw=1.2,
        label="in-place leaf append",
    )
    ax[1].plot(
        [x["N"] for x in r], [x["write_B_row"] for x in r], color="#d62728", lw=1.2
    )
    ax[1].plot(
        [x["N"] for x in r],
        [x["read_B_row"] for x in r],
        color="#d62728",
        lw=1.2,
        ls=":",
    )
for a in ax:
    a.set_xscale("log")
    a.axvline(256 * 2**20 / 112, color="gray", ls="--", lw=0.8)
    a.set_xlabel("rows added, N")
ax[0].set_yscale("log")
ax[0].set_ylabel(r"add time ($\mu$s/row)")
ax[0].legend(fontsize=7)
ax[1].set_ylabel("disk bytes per row (solid: write, dotted: read)")
ax[0].set_title("(a) add time", fontsize=9)
ax[1].set_title("(b) disk traffic", fontsize=9)
fig.tight_layout()
fig.savefig(OUT)
