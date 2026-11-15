import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA, OUT = Path(sys.argv[1]), sys.argv[2]
GRAY, INK, MUTED = "#6b6a63", "#2b2b28", "#8a897f"
COL = {
    "0": GRAY,
    "2": "#2a78d6",
    "2t0.1": "#eb6834",
    "2i": "#1baf7a",
    "2t0.03": "#eda100",
    "2t0.3": "#e87ba4",
}
LAB = {
    "0": "8-bit (default)",
    "2": "residual, re-read all",
    "2t0.03": r"residual, $\tau$=0.03",
    "2t0.1": r"residual, $\tau$=0.1",
    "2t0.3": r"residual, $\tau$=0.3",
    "2i": "residual, inherit anchor",
}
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e5df",
        "grid.linewidth": 0.6,
    }
)


def profile():
    rows = {}
    for line in (DATA / "profile.txt").read_text().splitlines():
        c4, inh = re.search(r"code4=(\d) inherit=(\d)", line).groups()
        tau = float(re.search(r"tau=([\d.e+]+)", line).group(1))
        rd = float(re.search(r"reads/row=([\d.]+)", line).group(1))
        sl = float(re.search(r"slack/radius p50=([\d.]+)", line).group(1))
        rows[(c4, inh, tau)] = (rd, sl)
    return rows


def ram():
    by = {}
    for o in map(json.loads, (DATA / "ram/ram_D100.jsonl").read_text().splitlines()):
        by.setdefault(o["advise"][3:], []).append((o["add_us_row"], o["q_us_med"]))
    return {a: np.array(v) for a, v in by.items()}


def cold(arm):
    out = {}
    for p in sorted((DATA / "cold_m5").glob(f"c4_D100_1G_a{arm}_r*.jsonl")):
        for o in map(json.loads, p.read_text().splitlines()):
            out.setdefault(o["n"], []).append(o)
    return out


fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.0))
P = profile()
taus = [0.0, 0.03, 0.1, 0.3, 1.0]
a = ax[0, 0]
a.plot(
    [P[("2", "0", t)][1] for t in taus],
    [P[("2", "0", t)][0] for t in taus],
    "-o",
    color=COL["2"],
    ms=4,
    lw=1.5,
    label=r"lazy re-read, $\tau$ sweep",
)
for t, (rd, sl) in zip(taus, [P[("2", "0", t)] for t in taus]):
    a.annotate(
        rf"$\tau$={t:g}",
        (sl, rd),
        xytext=(4, 3),
        textcoords="offset points",
        color=INK,
        fontsize=7,
    )
rd, sl = P[("2", "1", 0.0)]
a.plot([sl], [rd], "s", color=COL["2i"], ms=6, label="inherit anchor")
a.axvline(P[("0", "0", 0.0)][1], color=GRAY, ls="--", lw=1, label="8-bit slack")
a.set_xlabel("leaf slack / radius (median)")
a.set_ylabel("exact re-reads per added row")
a.set_title("(a) Re-reads trade against slack", loc="left", color=INK)
a.legend(frameon=False, fontsize=7)

R = ram()
a = ax[0, 1]
for arm in ["0", "2", "2t0.03", "2t0.1", "2t0.3"]:
    v = R[arm]
    m, se = v.mean(0), v.std(0, ddof=1) / np.sqrt(len(v))
    a.errorbar(
        m[0],
        m[1] / 1e3,
        xerr=se[0],
        yerr=se[1] / 1e3,
        fmt="o",
        ms=5,
        color=COL[arm],
        label=LAB[arm],
    )
a.set_xlabel(r"in-RAM add ($\mu$s/row)")
a.set_ylabel("in-RAM query (ms, median)")
a.set_title("(b) In RAM, N=1e5, 5 seeds", loc="left", color=INK)
a.legend(frameon=False, fontsize=6.5, loc="upper right")

base = cold("0")
for k, (a, field) in enumerate([(ax[1, 0], "cold"), (ax[1, 1], "add_us_row")]):
    for arm in ["2", "2t0.1", "2i"] if field == "cold" else ["0", "2", "2t0.1", "2i"]:
        C = cold(arm)
        ns = sorted(n for n in C if n in base and n >= (1 if field == "cold" else 100))
        if field == "cold":
            g, e = [], []
            for n in ns:
                lr = np.log(
                    np.concatenate([o["cold"] for o in C[n]])
                    / np.concatenate([o["cold"] for o in base[n]])
                )
                g.append(np.exp(lr.mean()))
                e.append(np.exp(lr.mean()) * lr.std(ddof=1) / np.sqrt(len(lr)))
            a.errorbar(
                ns,
                g,
                yerr=e,
                fmt="-o",
                ms=3.5,
                lw=1.5,
                color=COL[arm],
                label=LAB[arm],
                capsize=0,
            )
        else:
            y = [np.median([o["add_us_row"] for o in C[n]]) for n in ns]
            a.plot(ns, y, "-o", ms=3.5, lw=1.5, color=COL[arm], label=LAB[arm])
    a.set_xscale("log")
    a.set_xlabel("N")
    if field == "cold":
        a.axhline(1, color=GRAY, ls="--", lw=1)
        a.set_ylabel("cold query time / 8-bit")
        a.set_title("(c) Cold query, 5-dim D=100", loc="left", color=INK)
    else:
        a.set_ylabel(r"add ($\mu$s/row, median)")
        a.set_ylim(0, 50)
        a.set_title("(d) Adds in the cold harness", loc="left", color=INK)
    a.legend(frameon=False, fontsize=6.5)
fig.tight_layout()
fig.savefig(OUT)
fig.savefig(OUT.replace(".pdf", ".png"), dpi=150)
