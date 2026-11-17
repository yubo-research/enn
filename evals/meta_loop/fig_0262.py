import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {
    "g10": "#2a78d6",
    "bg10f": "#eb6834",
    "bg10t": "#1baf7a",
    "bg10t1k": "#6250d6",
    "bg10t300": "#e87ba4",
    "main": "#52514e",
}
LABELS = {
    "g10": "unbuffered graph arm (g10)",
    "bg10f": "buffers R=3000, whole-buffer scans (bg10f, 0261)",
    "bg10t": "buffers R=3000, tagged (bg10t, new)",
    "bg10t1k": "buffers R=1000, tagged (bg10t1k, new)",
    "bg10t300": "buffers R=300, tagged (bg10t300, new)",
    "main": "main, brute force in RAM (0260)",
}
CAP_PANELS = [
    ("add_us_row", "add time (us / row)"),
    ("q_us_med", "posterior query, median (us)"),
    ("wr_b_row", "bytes written per added row"),
    ("q_rd_ops", "disk read requests per query"),
]


def load(p):
    return [json.loads(x) for x in open(p) if x.startswith("{")] if p.exists() else []


def style(ax):
    ax.grid(alpha=0.3, which="both", lw=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def capped(reports, out):
    here = reports / "report-0262-data" / "cap256"
    series = [
        (a, here / f"capq_D10_256M_{a}.jsonl")
        for a in ("g10", "bg10f", "bg10t", "bg10t1k")
    ]
    series.append(
        ("main", reports / "report-0260-data" / "cap256" / "capq_D10_256M_main.jsonl")
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    for (key, title), ax in zip(CAP_PANELS, axes.flat):
        for arm, path in series:
            rs = [r for r in load(path) if r["n"] >= 1000 and key in r]
            if not rs:
                continue
            ax.plot(
                [r["n"] for r in rs],
                [max(r[key], 0.05) for r in rs],
                ":" if arm == "main" else "-",
                color=COLORS[arm],
                marker="o",
                ms=3,
                lw=2 if arm.startswith("bg10t") else 1.2,
                label=LABELS[arm],
            )
        ax.axvline(3e6, color="k", lw=0.6, alpha=0.4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("N (rows)")
        ax.set_title(title, fontsize=9)
        style(ax)
    axes[0, 0].legend(fontsize=6, loc="upper left")
    fig.suptitle(
        "256 MB cap, D=10, 1 CPU, EBS; row file 0.9x cap at N=3e6 (vertical line)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out)
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=80)


def recall_1e6(reports):
    q = reports / "report-0262-data"
    files = {
        "g10": q / "qual" / "d10_1e6" / "r10.jsonl",
        "bg10f": q / "qual_v2" / "d10_1e6" / "r22.jsonl",
        "bg10t": q / "qual" / "d10_1e6" / "r31.jsonl",
        "bg10t1k": q / "qual_v2" / "d10_1e6" / "r34.jsonl",
        "bg10t300": q / "qual_v2" / "d10_1e6" / "r35.jsonl",
    }
    out = {}
    for arm, path in files.items():
        rs = [r["recall_at_10"] for r in load(path) if r["seed"] < 2]
        out[arm] = sum(rs) / len(rs) if rs else float("nan")
    return out


def in_ram(reports, out):
    rows = load(reports / "report-0262-data" / "probe_1e6b.jsonl")
    rows += [
        r
        for r in load(reports / "report-0262-data" / "probe_1e6.jsonl")
        if r["arm"] == "bg10f"
    ]
    order = ["g10", "bg10f", "bg10t", "bg10t1k", "bg10t300"]
    by = {r["arm"]: r for r in rows}
    arms = [a for a in order if a in by]
    recall = recall_1e6(reports)
    for a in arms:
        by[a]["recall"] = recall.get(a, float("nan"))
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.6))
    for ax, (key, title, scale) in zip(
        axes,
        [
            ("q_us", "query (ms)", 1e-3),
            ("rows_q", "rows scored per query (k)", 1e-3),
            ("add_us", "add (us / row)", 1.0),
            ("recall", "recall@10 x100 (2 seeds)", 100.0),
        ],
    ):
        vals = [by[a][key] * scale for a in arms]
        ax.barh(range(len(arms)), vals, color=[COLORS[a] for a in arms], height=0.7)
        for i, v in enumerate(vals):
            ax.text(
                v, i, f" {v:.1f}" if v >= 1 else f" {v:.2f}", va="center", fontsize=7
            )
        ax.set_yticks(range(len(arms)))
        ax.set_yticklabels(arms if ax is axes[0] else [""] * len(arms), fontsize=7)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=9)
        style(ax)
        ax.set_xlim((85, 102) if key == "recall" else (0, max(vals) * 1.35))
        ax.tick_params(labelsize=7)
    fig.suptitle("In RAM (1 GB cap, 1 CPU), D=10, N=1e6, 300 queries", fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=80)


def main():
    reports = Path(sys.argv[1])
    capped(reports, sys.argv[2])
    in_ram(reports, sys.argv[3])


if __name__ == "__main__":
    main()
