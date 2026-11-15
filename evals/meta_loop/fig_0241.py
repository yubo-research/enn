import collections
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(sys.argv[1])
OUT = sys.argv[2]
INK, INK2 = "#0b0b0b", "#52514e"
ARMS = [
    ("0", "8-bit codes (default)", "#2a78d6", "o"),
    ("2i", "residual 4-bit, inherited anchor", "#eb6834", "^"),
    ("2", "residual 4-bit, exact re-read", "#1baf7a", "s"),
]
FILES = [
    ("tree_codes.bin", "codes", "#2a78d6"),
    ("tree_slots.bin", "slots", "#eb6834"),
    ("tree_meta.bin", "meta", "#1baf7a"),
    ("train_y.bin", "train_y", "#eda100"),
    ("train_x.bin", "train_x", "#6250d6"),
]
HOT = ("tree_codes.bin", "tree_slots.bin", "tree_meta.bin", "train_y.bin")
THRESH = (50.0, 100.0, 300.0)
CACHE_MIB = 200.0


def load(arm):
    f = R / "cap256" / f"bpa_D100_256M_c{arm}.jsonl"
    rows = [json.loads(s) for s in f.read_text().splitlines() if s.strip()]
    grid = [r for r in rows if not r.get("progress") and r["n"] >= 1e4]
    prog = [r for r in rows if r.get("progress")]
    first = prog[0]["n"] if prog else 1e99
    return sorted([r for r in grid if r["n"] < first] + prog, key=lambda r: r["n"])


def onset(rows, t):
    for r in rows:
        if r["add_us_row"] >= t:
            return r["n"]
    return None


def style(a, title):
    a.set_title(title, fontsize=8.5, loc="left", color=INK)
    a.grid(alpha=0.25, lw=0.5)
    a.spines[["top", "right"]].set_visible(False)
    a.tick_params(labelsize=7, colors=INK2)


fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.2))
ax = ax.ravel()
table = {}
for arm, label, c, m in ARMS:
    rows = load(arm)
    n = np.array([r["n"] for r in rows])
    us = np.array([r["add_us_row"] for r in rows])
    rd = np.array([r["rd_b_row"] for r in rows]) / 1024
    ax[0].plot(n, us, marker=m, ms=3, lw=1.5, color=c, label=label)
    ax[1].plot(n, np.maximum(rd, 0.01), marker=m, ms=3, lw=1.5, color=c)
    pre = [r for r in rows if "res_mib" in r and r["add_us_row"] < 100]
    hot = None
    if pre and arm != "2":
        last = pre[0]
        hot = sum(last["res_mib"].get(k, 0) for k in HOT) * 2**20 / last["n"]
    table[arm] = {
        "onset": {str(int(t)): onset(rows, t) for t in THRESH},
        "hot_B_row": hot,
        "pred_onset": CACHE_MIB * 2**20 / hot if hot else None,
        "plateau_median_us": float(np.median(us[us >= 300]))
        if (us >= 300).any()
        else None,
        "max_n": int(n[-1]),
    }
    if hot:
        ax[0].axvline(table[arm]["pred_onset"], color=c, lw=0.9, ls="--")
ax[0].text(
    0.02,
    0.97,
    "dashed: predicted onset\n= 200 MiB / hot bytes per row",
    transform=ax[0].transAxes,
    fontsize=6.5,
    va="top",
    color=INK2,
)
for i in (0, 1):
    ax[i].set_xscale("log")
    ax[i].set_yscale("log")
    ax[i].set_xlabel("N (rows)", fontsize=7.5, color=INK2)
style(ax[0], "(a) add time, us/row")
style(ax[1], "(b) disk read, KiB/row")
ax[0].legend(fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.86))

labels, bottoms = [], None
snap = {}
for arm, label, c, m in ARMS:
    rows = [r for r in load(arm) if "res_mib" in r]
    full = [
        r for r in rows if sum(r["res_mib"].values()) > 190 and r["add_us_row"] < 100
    ]
    s = full[-1] if full else min(rows, key=lambda r: r["n"])
    snap[arm] = s
y = np.arange(len(ARMS))
left = np.zeros(len(ARMS))
for k, lab, c in FILES:
    w = np.array([snap[a]["res_mib"].get(k, 0) for a, *_ in ARMS])
    ax[2].barh(
        y, w, left=left, color=c, height=0.6, label=lab, edgecolor="#fcfcfb", lw=1.5
    )
    left += w
ax[2].set_yticks(
    y, [f"{a} (N={snap[a]['n'] / 1e6:.2f}M)" for a, *_ in ARMS], fontsize=7
)
ax[2].invert_yaxis()
ax[2].set_xlabel(
    "resident MiB (fincore), last sample before onset", fontsize=7.5, color=INK2
)
ax[2].legend(
    fontsize=6.5, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.22)
)
style(ax[2], "(c) what fills the 256 MB cap")

for name, files, c, m in (
    ("EBS (store disk)", ["ebs", "ebs2"], "#2a78d6", "o"),
    ("local NVMe", ["nvme"], "#eb6834", "^"),
):
    d = collections.defaultdict(list)
    for f in files:
        for s in (R / "extent" / f"{f}.jsonl").read_text().splitlines():
            r = json.loads(s)
            d[r["kib"]].append(r["us_per_read"] / 1000)
    k = np.array(sorted(d))
    mu = np.array([np.mean(d[x]) for x in k])
    lo = np.array([min(d[x]) for x in k])
    hi = np.array([max(d[x]) for x in k])
    ax[3].fill_between(k, lo, hi, color=c, alpha=0.18, lw=0)
    ax[3].plot(k, mu, marker=m, ms=3, lw=1.5, color=c, label=name)
    table["extent_" + files[0]] = {int(x): round(float(v), 3) for x, v in zip(k, mu)}
ax[3].axvspan(50, 100, color="#999999", alpha=0.15, lw=0)
ax[3].text(70, 0.09, "1000-row\ncodes run", fontsize=6, ha="center", color=INK2)
ax[3].set_xscale("log", base=2)
ax[3].set_yscale("log")
ax[3].set_xlabel("contiguous read size (KiB)", fontsize=7.5, color=INK2)
ax[3].legend(fontsize=6.5, frameon=False, loc="upper left")
style(ax[3], "(d) one cold read, ms (min-max band)")
fig.tight_layout()
fig.savefig(OUT)
fig.savefig(str(OUT).replace(".pdf", ".png"), dpi=150)
print(json.dumps(table, indent=1))
