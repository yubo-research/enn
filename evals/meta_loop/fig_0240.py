import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(sys.argv[1])
ARMS = [
    ("b0", "no buffers", "#4a4a4a", "o"),
    ("b64", "buffers at every level (B=64)", "#d1495b", "s"),
    ("b64L", "buffers at leaf parents (B=64)", "#2e86ab", "^"),
]


def load(arm):
    f = R / "cap256" / f"bpa_D100_256M_{arm}.jsonl"
    if not f.exists():
        return np.empty((0, 4))
    rows = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    return np.array(
        [
            (r["n"], r["add_us_row"], r["rd_b_row"], r["wr_b_row"])
            for r in rows
            if r["n"] >= 1e3
        ]
    )


fig, ax = plt.subplots(1, 4, figsize=(13, 3.2))
for arm, label, c, m in ARMS:
    a = load(arm)
    if not len(a):
        continue
    a = a[np.argsort(a[:, 0])]
    ax[0].plot(a[:, 0], a[:, 1], marker=m, ms=3, lw=1.2, color=c, label=label)
    ax[1].plot(a[:, 0], a[:, 2] / 1024, marker=m, ms=3, lw=1.2, color=c)
    ax[2].plot(a[:, 0], a[:, 3] / 1024, marker=m, ms=3, lw=1.2, color=c)
for i, t in enumerate(
    ["add time (us/row)", "disk read (KiB/row)", "disk write (KiB/row)"]
):
    ax[i].set_xscale("log")
    ax[i].set_xlabel("N (rows)")
    ax[i].set_title(f"({'abc'[i]}) {t}", fontsize=9, loc="left")
    ax[i].grid(alpha=0.3, lw=0.5)
    ax[i].spines[["top", "right"]].set_visible(False)
ax[0].set_yscale("log")
ax[1].set_ylim(0, 12)
ax[2].set_ylim(0, 14)
ax[0].legend(fontsize=7, frameon=False)


def ram(name, arm):
    out = []
    for dim in (10, 100):
        f = R / f"{name}_D{dim}.jsonl"
        rs = [json.loads(line) for line in f.read_text().splitlines()]
        for key in ("add_us_row", "q_us_med"):
            a = {r["seed"]: r[key] for r in rs if r["node_buf"] == 0}
            b = {r["seed"]: r[key] for r in rs if r["node_buf"] == arm}
            q = np.array([b[i] / a[i] for i in sorted(set(a) & set(b))])
            out.append((q.mean(), q.std(ddof=1) / np.sqrt(len(q))))
    return np.array(out)


labels = ["add D=10", "query D=10", "add D=100", "query D=100"]
x = np.arange(4)
for j, (name, arm, c, lab) in enumerate(
    [("ram", 64, "#d1495b", "every level"), ("ramL", 64, "#2e86ab", "leaf parents")]
):
    v = ram(name, arm)
    ax[3].bar(x + (j - 0.5) * 0.38, v[:, 0], 0.38, yerr=v[:, 1], color=c, label=lab)
ax[3].axhline(1.0, color="#4a4a4a", lw=0.8)
ax[3].set_xticks(x, labels, fontsize=7, rotation=20)
ax[3].set_ylabel("time / no-buffer time")
ax[3].set_ylim(0.8, 1.75)
ax[3].set_title("(d) in RAM, N=1e5, 5 seeds", fontsize=9, loc="left")
ax[3].legend(fontsize=7, frameon=False)
ax[3].spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(R / "fig_0240.pdf")
fig.savefig(R / "fig_0240.png", dpi=110)
print("wrote", R / "fig_0240.pdf")
