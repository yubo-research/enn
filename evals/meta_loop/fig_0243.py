import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

C = ["#8c8c8c", "#eb6834", "#2a78d6", "#1baf7a", "#a35bd6"]
MK = ["o", "s", "^", "v", "D"]
NONE, LEAF, ALL, ALLNS, R300 = range(5)


def rows(p):
    p = Path(p)
    return [json.loads(l) for l in p.read_text().splitlines()] if p.is_file() else []


def style(ax, title, ylab, logy=True):
    ax.set_title(title, fontsize=8, loc="left")
    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("N (rows)", fontsize=7)
    ax.set_ylabel(ylab, fontsize=7)
    ax.tick_params(labelsize=6)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.grid(True, which="major", color="#e4e3df", lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=5.5, frameon=False, loc="upper left")


def line(ax, xs, ys, i, label):
    ax.plot(xs, ys, color=C[i], marker=MK[i], ms=2.5, lw=1.1, label=label)


def sweep(ax, arms, key, nmin=1000):
    for f, i, lab in arms:
        r = sorted(
            (o for o in rows(f) if o["n"] >= nmin and key in o), key=lambda o: o["n"]
        )
        r = [o for o in r if o[key] > 0]
        if r:
            line(ax, [o["n"] for o in r], [o[key] for o in r], i, lab)


def cap_arms(d42, d43):
    return [
        (d42 / "cap/bpa_D100_256M_pb0.jsonl", NONE, "no buffers (0242)"),
        (
            d42 / "cap_owner/bpa_D100_256M_pb1000.jsonl",
            LEAF,
            "leaf parents, R=1000 (0242)",
        ),
        (
            d43 / "cap256/bpa_D100_256M_pb1000as.jsonl",
            ALL,
            "every level, R=1000, staggered",
        ),
        (d43 / "cap256/bpa_D100_256M_pb1000a.jsonl", ALLNS, "every level, R=1000"),
        (
            d43 / "cap256/bpa_D100_256M_pb300as.jsonl",
            R300,
            "every level, R=300, staggered",
        ),
    ]


def diag_arms(d43):
    g = d43 / "diag64_3e6"
    return [
        (d43 / "diag64/diag_D100_64M_pb0.jsonl", NONE, "no buffers"),
        (g / "diag_D100_64M_pb1000.jsonl", LEAF, "leaf parents, R=1000"),
        (g / "diag_D100_64M_pb1000as.jsonl", ALL, "every level, staggered"),
        (g / "diag_D100_64M_pb1000a.jsonl", ALLNS, "every level"),
    ]


def cold(ax, d42, d43):
    arms = [
        (d43 / "cold/cold_D100_1G_pb0.jsonl", NONE, "no buffers"),
        (d42 / "cold/cold_D100_1G_pb1000.jsonl", LEAF, "leaf parents (0242)"),
        (d43 / "cold/cold_D100_1G_pb1000a.jsonl", ALL, "every level, R=1000"),
    ]
    for f, i, lab in arms:
        r = [o for o in rows(f) if o.get("cold_q_us")]
        if r:
            line(
                ax,
                [o["n"] for o in r],
                [np.median(o["cold_q_us"]) / 1e3 for o in r],
                i,
                lab,
            )
    style(ax, "(e) cold query, 1 GB, EBS, D=100", "ms / query")


def ratio(r, a, D, k):
    by0 = {o["seed"]: o[k] for o in r if o["advise"] == "pb_0" and o["D"] == D}
    v = [
        o[k] / by0[o["seed"]]
        for o in r
        if o["advise"] == a and o["D"] == D and o["seed"] in by0
    ]
    se = np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
    return (np.mean(v), se) if v else None


def ram(ax, d43):
    g = d43 / "ram"
    arms = [
        (rows(g / "ram.jsonl"), "pb_1000", LEAF, "leaf parents, R=1000"),
        (rows(g / "ram_a.jsonl"), "pb_1000a", ALL, "every level, R=1000"),
        (rows(g / "ram_300.jsonl"), "pb_300a", R300, "every level, R=300"),
    ]
    cells = [
        (D, k, n)
        for D in (10, 100)
        for k, n in (("add_us_row", "add"), ("q_us_med", "query"))
    ]
    for x, (D, k, _) in enumerate(cells):
        for j, (r, a, i, lab) in enumerate(arms):
            m = ratio(r, a, D, k)
            if m:
                ax.bar(
                    x + 0.27 * j,
                    m[0],
                    0.25,
                    yerr=m[1],
                    color=C[i],
                    label=lab if x == 0 else None,
                )
    ax.axhline(1, color="#555", lw=0.6)
    ax.set_ylim(0, 3.6)
    ax.set_xticks(
        [x + 0.27 for x in range(len(cells))],
        [f"{n}\nD={D}" for D, _, n in cells],
        fontsize=6,
    )
    ax.set_title("(f) in RAM, N=1e5, vs no buffers", fontsize=8, loc="left")
    ax.set_ylabel("time ratio", fontsize=7)
    ax.tick_params(labelsize=6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=5.5, frameon=False)


def main():
    d42, d43, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    fig, ax = plt.subplots(2, 3, figsize=(7.2, 4.8))
    sweep(ax[0, 0], cap_arms(d42, d43), "add_us_row")
    style(ax[0, 0], "(a) add time, 256 MB cap, EBS, D=100", "us / row")
    sweep(ax[0, 1], cap_arms(d42, d43), "rd_b_row")
    style(ax[0, 1], "(b) disk reads, 256 MB cap, EBS", "bytes / row")
    sweep(ax[0, 2], diag_arms(d43), "add_us_row")
    style(ax[0, 2], "(c) add time, 64 MB cap, NVMe", "us / row")
    sweep(ax[1, 0], diag_arms(d43), "majflt_per_row")
    style(ax[1, 0], "(d) major faults, 64 MB cap, NVMe", "faults / row")
    cold(ax[1, 1], d42, d43)
    ram(ax[1, 2], d43)
    fig.tight_layout()
    fig.savefig(out)
    if len(sys.argv) > 4:
        fig.savefig(sys.argv[4], dpi=90)


if __name__ == "__main__":
    main()
