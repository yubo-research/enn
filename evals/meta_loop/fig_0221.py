import glob
import json
import pathlib
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

D, OUT = sys.argv[1], sys.argv[2]
F, LR = 16, 48


def load(f):
    r = [json.loads(x) for x in pathlib.Path(f).read_text().splitlines()]
    return np.array([x["N"] for x in r], float), np.array([x["us_per_row"] for x in r])


def smooth(N, t, w=5):
    return N[w // 2 : len(N) - w // 2], np.convolve(t, np.ones(w) / w, "valid")


def scan_rows(cap, N, bounded=True):
    leaves = N / LR
    top = 1
    while F**top < leaves:
        top += 1
    return sum(min(cap, F**j * LR) if bounded else cap for j in range(1, top)) + cap


fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.7))
cols = {1024: "C0", 4096: "C1", 16384: "C2"}
for f in sorted(glob.glob(f"{D}/sweep/c*_f16/*.jsonl")):
    c = int(f.split("/")[-2].split("_")[0][1:])
    if c in cols:
        ax[0].plot(*smooth(*load(f)), "--", color=cols[c], lw=1)
for f in sorted(glob.glob(f"{D}/sub/sub_c*_f16/*.jsonl")):
    c = int(f.split("/")[-2].split("_")[1][1:])
    ax[0].plot(*smooth(*load(f)), color=cols[c], lw=1.2, label=f"buffer {c}")
ax[0].plot([], [], "k--", lw=1, label="unbounded (dashed)")
ax[0].set(
    xscale="log",
    yscale="log",
    xlabel="N (rows)",
    ylabel="add time (us/row)",
    title="(a) buffer size, 256M cap",
)
ax[0].legend(fontsize=5.5)
ax[0].grid(alpha=0.3)
for f in sorted(glob.glob(f"{D}/long_c*/*.jsonl")):
    c = int(f.split("/")[-2].split("_")[1][1:])
    if c == 1024:
        continue
    N, t = load(f)
    ax[1].plot(N, t, ".", ms=2, color=cols[c], alpha=0.5)
    m = N >= 3e6
    b, a = np.polyfit(np.log(N[m]), t[m], 1)
    x = np.geomspace(3e6, N.max(), 20)
    ax[1].plot(
        x,
        a + b * np.log(x),
        color=cols[c],
        lw=1.5,
        label=f"buf {c}: {a:.0f}+{b:.2f} lnN",
    )
ax[1].axvspan(2e6, 3e6, color="0.85", lw=0)
ax[1].text(2.1e6, 1, "store ~ cap", fontsize=5.5)
ax[1].set(
    xscale="log",
    xlabel="N (rows)",
    ylabel="add time (us/row)",
    title="(b) to N=1e8, fits N>=3e6",
    ylim=(0, 20),
)
ax[1].legend(fontsize=5.5)
ax[1].grid(alpha=0.3)
pts = []
for pat, lab, mk in [
    (f"{D}/sweep/c*_f16/*.jsonl", "unbounded", "s"),
    (f"{D}/sub/sub_c*_f16/*.jsonl", "bounded", "o"),
]:
    for f in sorted(glob.glob(pat)):
        c = int(f.split("/")[-2].replace("sub_", "").split("_")[0][1:])
        if c < 1024:
            continue
        N, t = load(f)
        m = (N > 1e7) & (N <= 2e7)
        if m.sum():
            ax[2].plot(
                scan_rows(c, 1.5e7, lab == "bounded"),
                t[m].mean(),
                mk,
                color=cols.get(c, "k"),
                label=f"{lab} {c}",
            )
ax[2].set(
    xscale="log",
    yscale="log",
    xlabel="buffered rows per query path",
    ylabel="add time, N=1-2e7 (us/row)",
    title="(c) add vs query cost",
)
ax[2].legend(fontsize=5.5)
ax[2].grid(alpha=0.3)
for a in ax:
    a.minorticks_off()
ax[0].set_yticks([2, 5, 10, 30])
ax[0].set_yticklabels(["2", "5", "10", "30"])
ax[2].set_yticks([4, 10, 30])
ax[2].set_yticklabels(["4", "10", "30"])
plt.tight_layout()
plt.savefig(OUT)
