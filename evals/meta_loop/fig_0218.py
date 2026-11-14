import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COL = {"64M": "#1b9e77", "256M": "#d95f02", "1G": "#7570b3"}
CAP_MB = {"64M": 64, "256M": 256, "1G": 1024}


def load(p):
    return (
        [json.loads(x) for x in Path(p).read_text().splitlines()]
        if Path(p).exists()
        else []
    )


def series(rows):
    r = sorted((x for x in rows if x["n"] >= 1e4), key=lambda x: x["n"])
    return [
        np.array([x[k] for x in r]) for k in ("n", "add_us_row", "rd_b_row", "wr_b_row")
    ]


def main():
    d, out = Path(sys.argv[1]), sys.argv[2]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for lab in ("64M", "256M", "1G"):
        rows = load(d / f"bpa_D100_{lab}.jsonl")
        if not rows:
            continue
        n, us, rd, wr = series(rows)
        ax[0].plot(n, us, "o-", ms=2, lw=1, color=COL[lab], label=f"cap {lab}")
        nstar = CAP_MB[lab]
        ax[1].plot(n / nstar, us, "o-", ms=2, lw=1, color=COL[lab], label=f"cap {lab}")
        ax[2].plot(
            n / nstar, rd + wr, "o-", ms=2, lw=1, color=COL[lab], label=f"cap {lab}"
        )
    ax[0].set(
        xscale="log", yscale="log", xlabel="N (rows)", ylabel=r"add time ($\mu$s/row)"
    )
    ax[1].set(xscale="log", yscale="log", xlabel="N / cap (rows per MB)")
    ax[2].set(
        xscale="log",
        yscale="log",
        xlabel="N / cap (rows per MB)",
        ylabel="disk bytes / add",
    )
    ax[0].legend(fontsize=6, frameon=False)
    for a, t in zip(ax, "abc"):
        a.set_title(t, loc="left", fontsize=8, fontweight="bold")
        a.tick_params(labelsize=6)
        a.xaxis.label.set_size(7)
        a.yaxis.label.set_size(7)
    fig.tight_layout()
    fig.savefig(out)
    leaf_fig(d, out.replace(".pdf", "_leaf.pdf"))


def leaf_fig(d, out):
    import csv

    rows = list(csv.DictReader((d / "leaf_table.csv").read_text().splitlines()))
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3))
    for D, c in (("10", "#1b9e77"), ("100", "#d95f02")):
        r = [x for x in rows if x["D"] == D]
        L = [float(x["L"]) for x in r]
        ax[0].plot(
            L, [float(x["recall"]) for x in r], "o-", ms=3, color=c, label=f"D={D}"
        )
        ax[1].errorbar(
            L,
            [float(x["dloglik"]) for x in r],
            [float(x["se"]) for x in r],
            fmt="o-",
            ms=3,
            color=c,
            capsize=2,
        )
        ax[2].plot(L, [float(x["query_ratio_vs_main"]) for x in r], "o-", ms=3, color=c)
    ax[1].axhline(0, color="k", lw=0.5)
    ax[2].axhline(1, color="k", lw=0.5, ls=":")
    for a, yl in zip(
        ax,
        ("recall@10", r"$\Delta$ loglik vs main (mean $\pm$ SE)", "query time / main"),
    ):
        a.set(xscale="log", xlabel="leaf budget coefficient L", ylabel=yl)
        a.tick_params(labelsize=6)
        a.xaxis.label.set_size(7)
        a.yaxis.label.set_size(7)
    ax[2].set_yscale("log")
    ax[0].legend(fontsize=6, frameon=False)
    for a, t in zip(ax, "abc"):
        a.set_title(t, loc="left", fontsize=8, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out)


if __name__ == "__main__":
    main()
