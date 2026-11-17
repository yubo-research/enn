import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "g10": "#2a78d6",
    "bg10t": "#1baf7a",
    "h300": "#eb6834",
    "t300": "#6250d6",
    "main": "#52514e",
}
LABELS = {
    "g10": "unbuffered graph arm g10 (0262)",
    "bg10t": "tagged buffers R=3000 bg10t",
    "h300": "re-homed tagged buffers R=300 h300 (new)",
    "t300": "tagged buffers R=300 t300 (control)",
    "main": "main, brute force in RAM (0260)",
}
CAP_PANELS = [
    ("add_us_row", "add time (us / row)"),
    ("q_us_med", "posterior query, median (us)"),
    ("wr_b_row", "bytes written per added row"),
    ("q_rd_ops", "disk read requests per query"),
]
DIAG_ARMS = [
    ("t3000", "tags R=3000"),
    ("h3000", "+ re-home R=3000"),
    ("h1000", "+ re-home R=1000"),
    ("t300", "tags R=300"),
    ("h300", "+ re-home R=300"),
]
PLACES = [
    ("leaf", "leaf", "#2a78d6"),
    ("upper_buf", "upper-level buffer", "#9a9893"),
    ("leafparent_buf", "leaf-parent buffer", "#eb6834"),
]


def load(p):
    return [json.loads(x) for x in open(p) if x.startswith("{")] if p.exists() else []


def style(ax):
    ax.grid(alpha=0.3, which="both", lw=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def misses_panel(ax, diag):
    by_arm = {r["arm"]: r for r in diag}
    xs = np.arange(len(DIAG_ARMS))
    bottom = np.zeros(len(DIAG_ARMS))
    for place, label, color in PLACES:
        h = np.array(
            [
                sum(c[-1] for c in by_arm[a]["cells"] if c[0] == place and c[4] == 0)
                if a in by_arm
                else 0
                for a, _ in DIAG_ARMS
            ]
        )
        ax.bar(xs, h, bottom=bottom, color=color, label=label, width=0.65)
        bottom += h
    for x, (a, _) in zip(xs, DIAG_ARMS):
        if a in by_arm:
            ax.text(
                x, bottom[x] + 4, f"{by_arm[a]['recall']:.3f}", ha="center", fontsize=7
            )
    ax.set_xticks(
        xs, [lbl for _, lbl in DIAG_ARMS], rotation=25, ha="right", fontsize=7
    )
    ax.set_ylabel("missed true neighbours (of 3000)")
    ax.set_title("(a) where missed neighbours sit; recall on top", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    style(ax)


def ecdf(ax, vals, label, color, ls="-"):
    v = np.sort(np.asarray(vals, dtype=float)) + 1
    ax.step(
        v,
        np.arange(1, len(v) + 1) / len(v),
        where="post",
        label=label,
        color=color,
        ls=ls,
        lw=1.3,
    )


def rank_panel(ax, rank_rec, audit_rec):
    au = np.array(audit_rec["audit"])
    late = au[au[:, 0] >= 400_000]
    ecdf(ax, late[:, 2], "push time: tree descent only", "#9a9893", "--")
    ecdf(ax, late[:, 1], "push time: after graph re-home", "#eb6834", "--")
    lp = [x for x in rank_rec["ranks"] if x[0] == "leafparent_buf"]
    ecdf(ax, [x[2] for x in lp if x[1] == 1], "query time: found", "#1baf7a")
    ecdf(ax, [x[2] for x in lp if x[1] == 0], "query time: missed", "#d03b3b")
    lf = [x for x in rank_rec["ranks"] if x[0] == "leaf"]
    ecdf(ax, [x[2] for x in lf], "query time: rows in leaves", "#2a78d6")
    ax.set_xscale("log")
    ax.set_xlabel("1 + owner-leaf rank by distance to row")
    ax.set_ylabel("fraction of rows")
    ax.set_title("(b) placement of buffered rows, R=3000, $10^6$", fontsize=8)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    style(ax)


def diag_figure(data, out):
    diag = load(data / "diag_1e6.jsonl")
    rank_rec = load(data / "diag_rank_1e6.jsonl")[-1]
    audit_rec = load(data / "diag_audit_1e6.jsonl")[-1]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    misses_panel(axes[0], diag)
    rank_panel(axes[1], rank_rec, audit_rec)
    fig.tight_layout()
    fig.savefig(out)


def capped(reports, out):
    here = reports / "report-0263-data" / "cap256"
    series = [
        ("h300", here / "capq_D10_256M_h300.jsonl"),
        ("bg10t", here / "capq_D10_256M_bg10t.jsonl"),
        ("t300", here / "capq_D10_256M_t300.jsonl"),
        ("g10", reports / "report-0262-data" / "cap256" / "capq_D10_256M_g10.jsonl"),
        ("main", reports / "report-0260-data" / "cap256" / "capq_D10_256M_main.jsonl"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    for (key, title), ax in zip(CAP_PANELS, axes.flat):
        for arm, path in series:
            rs = [r for r in load(path) if r["n"] >= 1000 and r.get(key, 0) > 0]
            if rs:
                ax.plot(
                    [r["n"] for r in rs],
                    [max(r[key], 1e-3) for r in rs],
                    "o-",
                    ms=3,
                    lw=1.3,
                    color=COLORS[arm],
                    label=LABELS[arm],
                )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("N (rows)")
        style(ax)
    axes[0, 0].legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    fig.savefig(out)


def main():
    reports = Path(sys.argv[1])
    data = reports / "report-0263-data"
    diag_figure(data, data / "fig_0263a.pdf")
    capped(reports, data / "fig_0263b.pdf")


if __name__ == "__main__":
    main()
