"""Build the tables and figure for iaml_3.tex from reports/mbpann/results.

Usage: python reports/iaml_3/make_iaml3.py
"""

import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "mbpann", "results")
MODELS = [
    ("bpann_disk", "BPANN\\_DISK (identity metric)"),
    ("mbpann_identity", "MBPANN\\_DISK, weights never set"),
    ("mbpann_learned", "MBPANN\\_DISK, learned weights"),
]


def read(name):
    with open(os.path.join(RES, name)) as f:
        return f.read()


def write(name, text):
    with open(os.path.join(HERE, name), "w") as f:
        f.write(text)


def pm(values, fmt):
    v = np.asarray(values)
    return f"${fmt.format(v.mean())} \\pm {fmt.format(v.std(ddof=1))}$"


def advantage_table(adv):
    seeds = sorted(adv, key=int)
    lines = []
    for key, label in MODELS:
        ll = [adv[s][key]["test_ll"] for s in seeds]
        rmse = [adv[s][key]["test_rmse"] for s in seeds]
        rec = [adv[s][key]["recall"] for s in seeds]
        per_seed = " / ".join(f"${x:.3f}$" for x in ll)
        lines.append(f"{label} & {per_seed} & {pm(ll, '{:.3f}')} & {pm(rmse, '{:.3f}')} & ${np.mean(rec):.2f}$\\\\")
    write("t_advantage.tex", "\n".join(lines) + "\n")


DIAG = re.compile(
    r"budget=\s*(\d+) beam=\s*(\d+) order=(\w+) sync_first=(\w+)\s+recall=([\d.]+) distinct_batches=([\d.]+)"
)
N_ROWS, ROWS_PER_FRAGMENT, N_FRAGMENTS = 40_000, 80_000, 4
TOPO = re.compile(r"all fragments, beam=(\d+): recall=([\d.]+) mean visited leaves/query=([\d.]+)")


def diag_rows(name):
    return [m.groups() for m in DIAG.finditer(read(name))]


def topo_rows(name):
    return [(int(b), float(r), float(v)) for b, r, v in TOPO.findall(read(name))]


def rootfix_table():
    before, after = diag_rows("bpann_recall_diag.log"), diag_rows("bpann_recall_diag_rootfix.log")
    lines = []
    for (bud, beam, order, sync, r0, d0), (_, _, _, _, r1, d1) in zip(before, after):
        if sync != "True" or order != "random":
            continue
        eff = min(max(N_ROWS // ROWS_PER_FRAGMENT, 2), N_FRAGMENTS, int(bud))
        lines.append(f"production path & {bud} ({eff}) & {beam} & ${r0}$ & ${r1}$ & ${d0}$ & ${d1}$\\\\")
    lines.append("\\midrule")
    for (beam, r0, v0), (_, r1, v1) in zip(topo_rows("bpann_topology_diag.log"), topo_rows("bpann_topology_diag_rootfix.log")):
        lines.append(f"all fragments & all & {beam} & ${r0:.3f}$ & ${r1:.3f}$ & \\multicolumn{{2}}{{c}}{{leaves/query: ${v0:.0f}$ vs ${v1:.0f}$}}\\\\")
    write("t_rootfix.tex", "\n".join(lines) + "\n")


def weights_panel(ax0, adv):
    seeds = sorted(adv, key=int)
    dims = np.arange(12)
    for j, s in enumerate(seeds):
        w = np.asarray(adv[s]["weights"])
        ax0.scatter(dims + (j - 1) * 0.18, w, s=14, color="0.2", marker="os^"[j], label=f"seed {s}")
    ax0.set_yscale("log")
    ax0.set_xticks(dims)
    ax0.set_xticklabels([f"$x_{{{d}}}$" for d in dims], fontsize=7)
    ax0.set_ylabel("learned weight $a_i$")
    ax0.set_title("(a) learned metric, two-of-twelve problem", fontsize=9, loc="left")
    ax0.axhline(1.0, color="0.7", lw=0.8, ls=":")
    ax0.text(11.4, 1.3, "identity", color="0.5", fontsize=7, ha="right")
    ax0.legend(frameon=False, fontsize=7, loc="center right")


def recall_panel(ax1):
    t0, t1 = topo_rows("bpann_topology_diag.log"), topo_rows("bpann_topology_diag_rootfix.log")
    beams = [b for b, _, _ in t0]
    ax1.plot(beams, [r for _, r, _ in t0], "o-", color="0.55", label="current code (wrong root)")
    ax1.plot(beams, [r for _, r, _ in t1], "s-", color="0.1", label="root = page 0 (temporary fix)")
    for (b, r, v) in t0:
        ax1.annotate(f"{v:.0f}", (b, r), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=7, color="0.45")
    for (b, r, v) in t1:
        ax1.annotate(f"{v:.0f}", (b, r), textcoords="offset points", xytext=(-4, 5), ha="right", fontsize=7)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(beams)
    ax1.set_xticklabels([str(b) for b in beams])
    ax1.set_xlim(0.8, 20)
    ax1.set_ylim(-0.12, 1.15)
    ax1.set_xlabel("beam width (all four fragments searched)")
    ax1.set_ylabel("recall@10")
    ax1.set_title("(b) BPANN recall vs beam width", fontsize=9, loc="left")
    ax1.legend(frameon=False, fontsize=7, loc="center right", title="labels: leaves visited per query", title_fontsize=7)


def figure(adv):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.0, 2.6), constrained_layout=True)
    weights_panel(ax0, adv)
    recall_panel(ax1)
    fig.savefig(os.path.join(HERE, "fig_iaml3.pdf"))
    fig.savefig(os.path.join(HERE, "fig_iaml3.png"), dpi=150)


def main():
    adv = json.loads(read("metric_advantage.json"))
    advantage_table(adv)
    rootfix_table()
    figure(adv)


if __name__ == "__main__":
    main()
