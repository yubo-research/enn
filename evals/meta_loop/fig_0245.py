import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import fig_0244
import matplotlib.pyplot as plt
import matplotlib.ticker
from fig_0244 import plot_run, rows, style

C = {"1000ap": "#1baf7a", "1000aph": "#a35bd6"}
LAB = {
    "1000ap": "every level + prefetch (0244)",
    "1000aph": "+ hole punch on take (0245)",
}

fig_0244.C, fig_0244.LAB = C, LAB


def stall(ax, d):
    for a, col in C.items():
        r = sorted(
            rows(d / f"stall_faults/stall_D100_256M_pb{a}.jsonl"), key=lambda o: o["n"]
        )
        if not r:
            continue
        n = [o["n"] for o in r]
        ax.plot(
            n,
            [o["rd_ops_row"] for o in r],
            "o-",
            color=col,
            lw=0.9,
            ms=2.5,
            label=f"{LAB[a].split(' (')[0]}: device reads",
        )
        ax.plot(
            n,
            [o["flt_push_row"] for o in r],
            "s:",
            color=col,
            lw=0.7,
            ms=2,
            label="  push faults (incl. holes)",
        )
    ax.set_xscale("log")
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(0, 0.17)
    ax.set_title("(c) per added row, stall probe", fontsize=8, loc="left")
    ax.set_xlabel("N (rows)", fontsize=7)
    ax.set_ylabel("events per added row", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(True, color="#e4e3df", lw=0.5)
    ax.axvline(1.2e6, color="#8c8c8c", lw=0.5, ls=":")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=5, frameon=False, loc="upper left")


def main():
    d = Path(sys.argv[1])
    runs = {a: sorted(d.glob(f"cap256/r*/bpa_D100_256M_pb{a}.jsonl")) for a in C}
    fig, ax = plt.subplots(
        1, 3, figsize=(7.0, 2.5), gridspec_kw={"width_ratios": [1.15, 1, 1]}
    )
    for a, files in runs.items():
        for i, f in enumerate(files):
            plot_run(ax, a, f, i == 0)
    style(ax[0], "(a) add time, 256 MB cap", "us per added row")
    style(ax[1], "(b) disk reads during adds", "bytes read per added row")
    ax[1].set_xlim(8e5, 1.2e7)
    stall(ax[2], d)
    fig.tight_layout()
    fig.savefig(d / "fig_0245.pdf")


if __name__ == "__main__":
    main()
