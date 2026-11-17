import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARMS = {
    "main": ("main (brute force)", "#7f7f7f"),
    "def": ("HEAD default tree walk, 100 lnN", "#d62728"),
    "g10": ("graph sweep + merged walk, 10 lnN", "#2ca02c"),
}
QCOL = {"0": "#d62728", "10": "#2ca02c", "30": "#1f77b4"}


def load(p):
    return [json.loads(x) for x in open(p)] if p.exists() else []


def fit(n, y, nmin=1e3, nmax=1e6):
    m = (n >= nmin) & (n <= nmax)
    if m.sum() < 3:
        return None
    a = np.vstack([np.ones(m.sum()), np.log(n[m])]).T
    coef = np.linalg.lstsq(a, y[m], rcond=None)[0]
    r = y[m] / (a @ coef)
    beyond = [
        (float(nn), float(yy / (coef[0] + coef[1] * np.log(nn))))
        for nn, yy in zip(n[~m & (n > nmax)], y[~m & (n > nmax)])
    ]
    return coef, float(r.max()), float(r.min()), beyond


def cap_panels(ax_add, ax_q, ax_ops, data):
    for arm, (lab, col) in ARMS.items():
        rs = load(data / "cap256" / f"capq_D10_256M_{arm}.jsonl")
        rs = [r for r in rs if r["n"] >= 1000]
        if not rs:
            continue
        n = np.array([r["n"] for r in rs], float)
        add = np.array([r["add_us_row"] for r in rs])
        q = np.array([r["q_us_med"] for r in rs]) / 1e3
        ops = np.array([r["q_rd_ops"] for r in rs])
        ax_add.plot(n, add, "o-", color=col, label=lab, ms=4)
        ax_q.plot(n, q, "o-", color=col, label=lab, ms=4)
        ax_ops.plot(n, np.maximum(ops, 0.1), "o-", color=col, label=lab, ms=4)
        for name, y in (("add_us", add), ("query_ms", q)):
            f = fit(n, y)
            if f:
                (a, b), hi, lo, beyond = f
                st = " ".join(f"N={nn:.0e}: {rr:.1f}x fit" for nn, rr in beyond)
                print(
                    f"{arm:5s} {name:9s} fit 1e3..1e6 a={a:9.3f} b={b:8.3f} in-range ratio {lo:.2f}..{hi:.2f} | {st}"
                )
    for ax, t in (
        (ax_add, "(a) add, us/row"),
        (ax_q, "(b) posterior, ms/query (median)"),
        (ax_ops, "(c) disk reads/query"),
    ):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("N")
        ax.set_title(t, fontsize=9, loc="left")
        ax.grid(alpha=0.3, which="both")


def qual_panel(ax, data):
    p = data / "qual" / "stats.json"
    if not p.exists():
        return
    rows = json.loads(p.read_text())
    names = {"0": "def", "10": "g10", "30": "g30"}
    xs, labels = [], []
    for i, case in enumerate(("d10_1e5", "d10_1e6")):
        for j, arm in enumerate(("0", "30", "10")):
            r = next((r for r in rows if r["case"] == case and r["arm"] == arm), None)
            if not r or "dloglik_vs_main" not in r:
                continue
            _, m, se, pv = r["dloglik_vs_main"]
            x = i * 4 + j
            col = QCOL[arm]
            ax.errorbar(x, m, yerr=1.96 * se, fmt="o", color=col)
            ax.annotate(
                f"p={pv:.2f}",
                (x, m),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=6,
            )
            xs.append(x)
            labels.append(f"{names[arm]}\n{case[4:]}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_title(
        "(d) loglik minus main, paired 10 seeds (95% CI)", fontsize=9, loc="left"
    )
    ax.grid(alpha=0.3, axis="y")


def main():
    data, out = Path(sys.argv[1]), sys.argv[2]
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.6))
    cap_panels(axs[0, 0], axs[0, 1], axs[1, 0], data)
    qual_panel(axs[1, 1], data)
    axs[0, 0].legend(fontsize=6, loc="upper left")
    fig.tight_layout()
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=150)


if __name__ == "__main__":
    main()
