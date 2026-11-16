import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

B, O, GR = "#2a78d6", "#eb6834", "#8a8984"
FIT_LO, FIT_HI = 1e3, 1e6
D = Path(sys.argv[1])


def jl(p):
    p = Path(p)
    if not p.exists():
        return []
    return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]


def style(ax, title, ylab, logx=True, xlab="N (rows)"):
    ax.set(xscale="log" if logx else "linear", xlabel=xlab, ylabel=ylab)
    ax.set_title(title, fontsize=8, loc="left")
    ax.xaxis.label.set_size(7)
    ax.yaxis.label.set_size(7)
    ax.tick_params(labelsize=6)
    ax.grid(True, color="#e4e3df", lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def reps(arm):
    by = {}
    for f in sorted(glob.glob(str(D / f"pace256*/rep*/stall_D100_256M_pb{arm}.jsonl"))):
        for o in jl(f):
            by.setdefault(o["n"], []).append(o)
    n = sorted(by)
    return n, by


def panel_a(a):
    for arm, c, lb in (("3000aph_0", B, "unpaced (HEAD)"), ("3000aph_1", O, "paced")):
        n, by = reps(arm)
        w = [np.mean([o["wall_us_row"] for o in by[k]]) for k in n]
        lo = [min(o["wall_us_row"] for o in by[k]) for k in n]
        hi = [max(o["wall_us_row"] for o in by[k]) for k in n]
        cpu = [np.mean([o["cpu_us_row"] for o in by[k]]) for k in n]
        a.plot(n, w, "-o", color=c, ms=3, lw=1.5, label=f"{lb}: wall")
        a.fill_between(n, lo, hi, color=c, alpha=0.2, lw=0)
        a.plot(n, cpu, ":", color=c, lw=1.2, label=f"{lb}: CPU")
    style(
        a,
        "(a) Add time, 256 MB cap, R=3000, 4 runs/arm",
        "us/row (1e6-row windows)",
        logx=False,
    )
    a.set_ylim(bottom=0)
    a.set_xlabel("N (rows)", fontsize=7)
    a.legend(fontsize=5.5)


def dirty_means(arm):
    groups = {
        "train_x+y": ("train_",),
        "tree_pbuf": ("index/tree_pbuf",),
        "tree_codes": ("index/tree_codes",),
        "other index": ("index/",),
    }
    acc = {g: [] for g in groups}
    for f in sorted(glob.glob(str(D / f"pace256*/rep*/dirty_{arm}.jsonl"))):
        for r in jl(f):
            fs, used = r["files"], set()
            for g, pre in groups.items():
                ks = [k for k in fs if k.startswith(pre) and k not in used]
                used.update(ks)
                acc[g].append(sum(fs[k][1] for k in ks))
    return {g: float(np.mean(v)) if v else 0.0 for g, v in acc.items()}


def panel_b(a):
    cols = [B, O, "#1baf7a", GR]
    arms = (("3000aph_0", "unpaced"), ("3000aph_1", "paced"))
    bottoms = np.zeros(len(arms))
    means = [dirty_means(x) for x, _ in arms]
    for i, g in enumerate(means[0]):
        v = np.array([m[g] for m in means])
        a.bar(
            [lb for _, lb in arms], v, bottom=bottoms, color=cols[i], width=0.5, label=g
        )
        bottoms += v
    style(
        a,
        "(b) Mean dirty page cache by file (256 MB, 1 Hz)",
        "MiB",
        logx=False,
        xlab="",
    )
    a.legend(fontsize=5.5)


def fit(n, y):
    n, y = np.asarray(n, float), np.asarray(y, float)
    m = (n >= FIT_LO) & (n <= FIT_HI)
    b, a = np.polyfit(np.log(n[m]), y[m], 1)
    return a, b


def panel_c(a, out):
    base = D.parent / "report-0248-data/cap1G/bpa_D100_1G_pb3000aph.jsonl"
    for f, c, lb, ls in (
        (D / "cap1G/bpa_D100_1G_pb3000aphu.jsonl", B, "unpaced, run 1", "-"),
        (D / "cap1G/r2/bpa_D100_1G_pb3000aphu.jsonl", B, "unpaced, run 2", "-."),
        (D / "cap1G/bpa_D100_1G_pb3000aph.jsonl", O, "paced, run 1", "-"),
        (D / "cap1G/r2/bpa_D100_1G_pb3000aph.jsonl", O, "paced, run 2", "-."),
        (base, GR, "unpaced (0248)", ":"),
    ):
        g = [o for o in jl(f) if not o.get("progress") and o.get("rows", 0) > 0]
        if not g:
            continue
        n = [o["n"] for o in g]
        y = [o["add_us_row"] for o in g]
        a.plot(n, y, ls, color=c, marker="o", ms=2.5, lw=1.3, label=lb)
        fa, fb = fit(n, y)
        xs = np.logspace(3, np.log10(max(n)), 50)
        a.plot(xs, fa + fb * np.log(xs), "--", color=c, lw=0.6)
        out.append(f"{lb}: add us/row = {fa:.2f} + {fb:.3f} lnN (fit 1e3..1e6)")
        for o in g:
            if o["n"] > FIT_HI:
                p = fa + fb * np.log(o["n"])
                out.append(
                    f"  N={o['n']:.0e} meas {o['add_us_row']:.1f} fit {p:.1f} ratio {o['add_us_row'] / p:.2f}"
                )
    style(a, "(c) 1 GB cap to 3e7, R=3000; dashed = lnN fit", "us/row")
    a.set_ylim(bottom=0)
    a.set_xlabel("N (rows)", fontsize=7)
    a.legend(fontsize=5.5)


def panel_d(a):
    rows = jl(D / "ram/ram.jsonl")
    by = {(r["advise"], r["seed"], r["D"]): r for r in rows}
    labels, vals, ses = [], [], []
    for d, base, arm in (
        (100, "pb_0u", "pb_0"),
        (100, "pb_3000aphu", "pb_3000aph"),
        (10, "pb_0u", "pb_0"),
    ):
        for k, kl in (("add_us_row", "add"), ("q_us_med", "query")):
            ss = sorted(
                s for (x, s, dd) in by if x == base and dd == d and (arm, s, d) in by
            )
            r = np.array([by[(arm, s, d)][k] / by[(base, s, d)][k] for s in ss])
            if len(r) == 0:
                continue
            labels.append(f"D={d}\n{'R=3000' if '3000' in arm else 'no buf'}\n{kl}")
            vals.append(r.mean())
            ses.append(r.std(ddof=1) / np.sqrt(len(r)))
    x = np.arange(len(vals))
    a.bar(
        x,
        vals,
        yerr=ses,
        color=[O if "add" in l else B for l in labels],
        width=0.6,
        capsize=2,
    )
    a.axhline(1.0, color="k", lw=0.6)
    a.set_xticks(x, labels, fontsize=5)
    style(
        a,
        "(d) In RAM (1 GB, N=1e5, 5 seeds): paced / unpaced",
        "ratio (mean +- SE)",
        logx=False,
        xlab="",
    )
    a.set_ylim(0.7, 1.15)


def panel_e(a):
    for arm, c, lb in (("3000aph_0", B, "unpaced"), ("3000aph_1", O, "paced")):
        n, by = reps(arm)
        a.plot(
            n,
            [np.mean([o["wall_us_row"] for o in by[k]]) for k in n],
            "-o",
            color=c,
            ms=3,
            lw=1.5,
            label=f"{lb}, no sampler (4 runs)",
        )
    for f, c, lb in (
        (D / "wchan256/stall_D100_256M_pb3000aph_0.jsonl", B, "unpaced + sampler"),
        (D / "wchan256/stall_D100_256M_pb3000aph_1.jsonl", O, "paced + sampler"),
        (
            D / "wchan256b/stall_D100_256M_pb3000aph_1.jsonl",
            O,
            "paced + sampler (repeat)",
        ),
    ):
        g = jl(f)
        a.plot(
            [o["n"] for o in g],
            [o["wall_us_row"] for o in g],
            "--x",
            color=c,
            ms=3,
            lw=1.0,
            label=lb,
        )
    style(a, "(e) A 200 Hz /proc wchan sampler slows the run", "us/row", logx=False)
    a.set_ylim(bottom=0)
    a.set_xlabel("N (rows)", fontsize=7)
    a.legend(fontsize=5)


def panel_f(a):
    v = []
    for p in (0, 1):
        r = json.loads((D / f"residency/pace{p}.json").read_text())
        v.append(100.0 * r["old_resident_pages"] / r["old_pages"])
    a.bar(
        ["unpaced\n(unaligned drops)", "paced\n(2 MiB-aligned drops)"],
        v,
        color=[B, O],
        width=0.5,
    )
    for i, x in enumerate(v):
        a.text(i, x, f"{x:.2f}%", ha="center", va="bottom", fontsize=6)
    style(
        a,
        "(f) Dropped log pages still cached (2e5 rows)",
        "% of old train_x pages",
        logx=False,
        xlab="",
    )


def main():
    out = []
    fig, ax = plt.subplots(2, 3, figsize=(10.0, 5.6))
    panel_a(ax[0, 0])
    panel_b(ax[0, 1])
    panel_c(ax[0, 2], out)
    panel_d(ax[1, 0])
    panel_e(ax[1, 1])
    panel_f(ax[1, 2])
    fig.tight_layout()
    fig.savefig(sys.argv[2])
    (D / "fit_0249.txt").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
