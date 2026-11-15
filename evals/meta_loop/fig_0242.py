import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

C = ["#2a78d6", "#eb6834", "#1baf7a"]
MK = ["o", "s", "^"]


def rows(p):
    p = Path(p)
    return [json.loads(l) for l in p.read_text().splitlines()] if p.is_file() else []


def style(ax, title, ylab):
    ax.set_title(title, fontsize=8, loc="left")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N (rows)", fontsize=7)
    ax.set_ylabel(ylab, fontsize=7)
    ax.tick_params(labelsize=6)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.grid(True, which="major", color="#e4e3df", lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def line(ax, xs, ys, i, label):
    ax.plot(xs, ys, color=C[i], marker=MK[i], ms=3, lw=1.5, label=label)


def counts(ax, d):
    sets = [
        ("pc_D10_i0", "D=10 uniform"),
        ("pc_D100_i0", "D=100 uniform"),
        ("pc_D100_i5", "D=100, 5-dim"),
    ]
    for i, (f, lab) in enumerate(sets):
        r = [o for o in rows(d / f"{f}.jsonl") if o["n"] >= 100]
        line(ax, [o["n"] for o in r], [o["leaf_parent_expands"][0] for o in r], i, lab)
    style(ax, "(a) leaf-parent nodes expanded per query", "nodes / query")
    ax.legend(fontsize=6, frameon=False, loc="best")


def cap(ax, d):
    arms = [
        ("cap/bpa_D100_256M_pb0.jsonl", "no buffers"),
        ("cap/bpa_D100_256M_pb1000.jsonl", "R=1000, replay from root"),
        ("cap_owner/bpa_D100_256M_pb1000.jsonl", "R=1000, replay in owner"),
    ]
    for i, (f, lab) in enumerate(arms):
        r = sorted((o for o in rows(d / f) if o["n"] >= 1000), key=lambda o: o["n"])
        if r:
            line(ax, [o["n"] for o in r], [o["add_us_row"] for o in r], i, lab)
    style(ax, "(b) add time, 256 MB cap, EBS, D=100", "us / row")
    ax.legend(fontsize=6, frameon=False, loc="best")


def diag(ax, d, key, title, ylab):
    arms = [
        ("diag/diag_D100_64M_pb0.jsonl", "no buffers"),
        ("diag/diag_D100_64M_pb1000r.jsonl", "R=1000, replay from root"),
        ("diag/diag_D100_64M_pb1000.jsonl", "R=1000, replay in owner"),
    ]
    for i, (f, lab) in enumerate(arms):
        r = [o for o in rows(d / f) if o.get("pbuf", 0) or key != "leaves_per_flush"]
        r = [o for o in r if key in o and o[key] > 0]
        if r:
            line(ax, [o["n"] for o in r], [o[key] for o in r], i, lab)
    style(ax, title, ylab)
    ax.legend(fontsize=6, frameon=False, loc="best")


def cold(ax, d):
    import numpy as np

    for i, (f, lab) in enumerate(
        [
            ("cold/cold_D100_1G_pb0.jsonl", "no buffers"),
            ("cold/cold_D100_1G_pb1000.jsonl", "R=1000, replay in owner"),
        ]
    ):
        r = rows(d / f)
        if r:
            line(
                ax,
                [o["n"] for o in r],
                [np.median(o["cold_q_us"]) / 1e3 for o in r],
                [0, 2][i],
                lab,
            )
    style(ax, "(f) cold query, 1 GB, EBS, D=100", "ms / query")
    ax.legend(fontsize=6, frameon=False, loc="best")


def main():
    d, out = Path(sys.argv[1]), sys.argv[2]
    fig, ax = plt.subplots(2, 3, figsize=(7.2, 4.6))
    counts(ax[0, 0], d)
    cap(ax[0, 1], d)
    diag(ax[0, 2], d, "rd_b_row", "(c) disk reads, 64 MB cap, NVMe", "bytes / row")
    diag(
        ax[1, 0],
        d,
        "leaves_per_flush",
        "(d) leaves written per flush",
        "leaves / flush",
    )
    diag(ax[1, 1], d, "add_us_row", "(e) add time, 64 MB cap, NVMe", "us / row")
    cold(ax[1, 2], d)
    fig.tight_layout()
    fig.savefig(out)
    if len(sys.argv) > 3:
        fig.savefig(sys.argv[3], dpi=90)


if __name__ == "__main__":
    main()
