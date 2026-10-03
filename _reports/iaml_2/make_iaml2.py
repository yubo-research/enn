"""Build the tables and figure for iaml_2.tex from reports/mbpann/results.

Usage: python reports/iaml_2/make_iaml2.py
"""

import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "mbpann", "results")
METHODS = [
    ("frozen_identity", "frozen identity"),
    ("oracle_final_metric", "oracle (final metric)"),
    ("bpann_copy_rebuild", "BPANN\\_DISK copy"),
    ("mbpann_rebuild", "MBPANN rebuild"),
    ("mbpann_auto", "MBPANN auto"),
    ("mbpann_rescale", "MBPANN rescale"),
]
STRATEGY = {
    "rescale": "rescale",
    "rebuild_in_place": "rebuild in place",
    "bpann_copy": "BPANN\\_DISK copy",
}


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def final_groups(rows):
    n = max(r["n"] for r in rows)
    g = defaultdict(lambda: defaultdict(list))
    for r in (r for r in rows if r["n"] == n):
        numeric = {k: v for k, v in r.items() if isinstance(v, (int, float))}
        cell = g[(r["problem"], r["method"])]
        for k, v in numeric.items():
            cell[k].append(v)
    return g


def pm(v, fmt="{:.3f}"):
    v = np.asarray(v, dtype=float)
    return f"${fmt.format(v.mean())} \\pm {fmt.format(v.std())}$"


def stream_table(rows):
    g = final_groups(rows)
    out = []
    for prob in ("sparse", "friedman"):
        for key, label in METHODS:
            v = g[(prob, key)]
            reb = f"{np.mean(v['rebuilds']):.0f}" if "rebuilds" in v else "--"
            cells = [
                prob if key == METHODS[0][0] else "",
                label,
                pm(v["loo"]),
                pm(v["test"]),
                f"{np.mean(v['loo_exact']):.3f} / {np.mean(v['test_exact']):.3f}",
                f"{np.mean(v['recall']):.2f}",
                reb,
                f"{np.mean(v['metric_sec']):.2f}",
                pm(v["query_sec"], "{:.2f}"),
            ]
            out.append(" & ".join(cells) + r" \\")
        if prob == "sparse":
            out.append(r"\midrule")
    return "\n".join(out) + "\n"


def timing_table(rows):
    out = []
    for batch in (500, 2000):
        for r in sorted((r for r in rows if r["batch"] == batch), key=lambda r: r["n"]):
            out.append(
                f"{batch} & {r['n']:,} & {STRATEGY[r['how']]} & {1e3 * r['change_sec']:.1f} & "
                f"{r['query_sec']:.2f} & {r['recall']:.2f} \\\\"
            )
        if batch == 500:
            out.append(r"\midrule")
    return "\n".join(out) + "\n"


def figure(timing, b500, b2000):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9.0, 3.4))
    colors = {"rescale": "C0", "rebuild_in_place": "C1", "bpann_copy": "C3"}
    for how, c in colors.items():
        for batch, ls, mk in ((500, "-", "o"), (2000, "--", "s")):
            pts = sorted(
                (r["n"], 1e3 * r["change_sec"])
                for r in timing
                if r["how"] == how and r["batch"] == batch
            )
            if pts:
                n, t = zip(*pts)
                ax0.plot(n, t, ls, marker=mk, color=c)
    handles = [
        plt.Line2D([], [], color=c, label=STRATEGY[how].replace(chr(92), ""))
        for how, c in colors.items()
    ]
    handles += [
        plt.Line2D([], [], color="0.3", ls="-", marker="o", label="batch 500"),
        plt.Line2D([], [], color="0.3", ls="--", marker="s", label="batch 2000"),
    ]
    ax0.set(
        xscale="log",
        yscale="log",
        xlabel="rows in index, $n$",
        ylabel="time for one metric change (ms)",
    )
    ax0.set_ylim(5e-2, 3e4)
    ax0.set_title("(a) Cost of one metric change", fontsize=10)
    ax0.legend(handles=handles, fontsize=6.5, frameon=False, ncol=2, loc="upper left")

    marks = {"sparse": "o", "friedman": "^"}
    for rows, fill in ((b500, True), (b2000, False)):
        n = max(r["n"] for r in rows)
        for i, (key, label) in enumerate(METHODS[1:]):
            for prob, mk in marks.items():
                sel = [
                    r
                    for r in rows
                    if r["n"] == n and r["method"] == key and r["problem"] == prob
                ]
                gap = [r["test"] - r["test_exact"] for r in sel]
                rec = [r["recall"] for r in sel]
                ax1.scatter(
                    rec,
                    gap,
                    marker=mk,
                    s=28,
                    color=f"C{i}",
                    facecolors=f"C{i}" if fill else "none",
                    label=label.replace("\\", "")
                    if (fill and prob == "sparse")
                    else None,
                )
    ax1.set(
        xscale="log",
        xlabel="recall@10 of the index",
        ylabel="test LL minus exact-neighbor test LL",
    )
    ax1.set_title("(b) Likelihood loss versus recall, $n=40{,}000$", fontsize=10)
    ax1.legend(fontsize=6.5, frameon=False, loc="lower right")
    ax1.axhline(0, color="0.7", lw=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_mbpann.pdf"))


def main():
    b500, b2000, timing = load("b500.json"), load("b2000.json"), load("timing.json")
    for name, text in (
        ("t_b500.tex", stream_table(b500)),
        ("t_b2000.tex", stream_table(b2000)),
        ("t_timing.tex", timing_table(timing)),
    ):
        with open(os.path.join(HERE, name), "w") as f:
            f.write(text)
        print(f"== {name}\n{text}")
    figure(timing, b500, b2000)


if __name__ == "__main__":
    main()
