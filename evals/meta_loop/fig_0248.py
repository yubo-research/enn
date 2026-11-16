import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

B, O, A, Y, GR = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8984"
D = Path(sys.argv[1])


def jl(p):
    p = Path(p)
    return (
        [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
        if p.exists()
        else []
    )


def style(ax, title, ylab, logx=True):
    if logx:
        ax.set_xscale("log")
    ax.set_title(title, fontsize=8, loc="left")
    ax.set_ylabel(ylab, fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(True, color="#e4e3df", lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.2))
a = ax[0, 0]
w = json.loads((D / "wchan256/wchan_1000aph.json").read_text())
L = jl(D / "wchan256/stall_D100_256M_pb1000aph.jsonl")
cats = [
    ("running", B, lambda k: k.startswith("R")),
    ("dirty throttle", O, lambda k: "balance_dirty" in k),
    ("page I/O wait", A, lambda k: "folio_wait" in k),
    ("mmap/munmap", Y, lambda k: "mmap" in k),
]
bins = sorted(w, key=int)
xs = np.arange(len(bins))
bottom = np.zeros(len(bins))
for name, c, f in cats + [("other", GR, None)]:
    v = []
    for b in bins:
        cnt = w[b]
        tot = sum(cnt.values())
        if f is None:
            s = sum(x for k, x in cnt.items() if not any(g(k) for _, _, g in cats))
        else:
            s = sum(x for k, x in cnt.items() if f(k))
        v.append(s / tot)
    a.bar(xs, v, bottom=bottom, color=c, width=0.8, label=name, edgecolor="white", lw=1)
    bottom += np.array(v)
lab = [
    f"{L[int(b)]['n']:.0e}".replace("e+0", "e") if int(b) < len(L) else "" for b in bins
]
a.set_xticks(xs, lab, fontsize=6)
style(a, "(a) Where wall time goes, R=1000 (256 MB)", "fraction of samples", logx=False)
a.set_xlabel("window end N (rows)", fontsize=7)
a.legend(
    fontsize=5.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False
)
a = ax[0, 1]
for arm, c in (("1000aph", B), ("3000aph", O)):
    for r, ls in (("rep1", "-"), ("rep2", "--")):
        L = jl(D / f"rsize256/{r}/stall_D100_256M_pb{arm}.jsonl")
        a.plot(
            [o["n"] for o in L],
            [o["wall_us_row"] for o in L],
            ls,
            color=c,
            lw=1.5,
            marker="o",
            ms=3,
            label=f"R={arm[:4]} {r}",
        )
style(a, "(b) Add time, 256 MB cap, D=100", "us/row (1e6-row windows)")
a.set_xlabel("N (rows)", fontsize=7)
a.legend(fontsize=5.5)
a = ax[1, 0]
L = jl(D / "diag256/stall_D100_256M_pb1000aph.jsonl")
n = [o["n"] for o in L]
for k, c, lb in (
    ("rd_ops_row", "#333333", "device read ops"),
    ("take_cold_pg_row", B, "cold pages at take"),
    ("tail_cold_row", O, "cold tail at flush"),
    ("meta_cold_row", A, "cold child meta"),
    ("push_cold_row", Y, "push into cold page"),
):
    a.plot(n, [o[k] for o in L], "-o", color=c, ms=3, lw=1.5, label=lb)
style(a, "(c) Read attribution, R=1000 (256 MB, probes on)", "per row")
a.set_xlabel("N (rows)", fontsize=7)
a.legend(fontsize=5.5)
a = ax[1, 1]
base = (
    Path(sys.argv[1]).parent / "report-0246-data/cap1G/r1/bpa_D100_1G_pb1000aph.jsonl"
)
for f, c, lb, ls in (
    (D / "cap1G/bpa_D100_1G_pb3000aph.jsonl", O, "R=3000 (this report)", "-"),
    (D / "cap1G/bpa_D100_1G_pb1000aph.jsonl", B, "R=1000 (this report)", "-"),
    (base, GR, "R=1000 (0246)", ":"),
):
    L = [o for o in jl(f) if not o.get("progress") and o.get("rows", 0) > 0]
    if L:
        a.plot(
            [o["n"] for o in L],
            [o["add_us_row"] for o in L],
            ls,
            color=c,
            marker="o",
            ms=3,
            lw=1.5,
            label=lb,
        )
style(a, "(d) Add time, 1 GB cap, D=100, to 3e7", "us/row")
a.set_xlabel("N (rows)", fontsize=7)
a.legend(fontsize=5.5)
fig.tight_layout()
fig.savefig(sys.argv[2])
