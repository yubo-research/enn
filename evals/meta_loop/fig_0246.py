import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C = {"1000aph": "#a35bd6", "0": "#d1495b"}
LAB = {"1000aph": "every-level buffers (1000aph)", "0": "no buffers (HEAD default)"}
FIT_LO, FIT_HI = 1e3, 1e6


def rows(f):
    f = Path(f)
    if not f.exists():
        return []
    return [json.loads(s) for s in f.read_text().splitlines() if s.strip()]


def grid(r):
    return [o for o in r if not o.get("progress") and o["rows"] > 0]


def fit(n, y):
    n, y = np.asarray(n, float), np.asarray(y, float)
    m = (n >= FIT_LO) & (n <= FIT_HI)
    b, a = np.polyfit(np.log(n[m]), y[m], 1)
    return a, b


def style(ax, title, ylab):
    ax.set_xscale("log")
    ax.set_title(title, fontsize=8, loc="left")
    ax.set_xlabel("N (rows)", fontsize=7)
    ax.set_ylabel(ylab, fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(True, color="#e4e3df", lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def scatter(ax, p, key, col):
    ax.plot(
        [o["n"] for o in p], [key(o) for o in p], ".", color=col, ms=1.2, alpha=0.35
    )


def fit_lines(ax, f, a, col, g, out):
    n = [o["n"] for o in g]
    y = [o["add_us_row"] for o in g]
    if sum(FIT_LO <= v <= FIT_HI for v in n) < 3:
        return
    fa, fb = fit(n, y)
    xs = np.logspace(3, np.log10(max(n)), 50)
    ax.plot(xs, fa + fb * np.log(xs), "--", color=col, lw=0.6)
    out.append(
        f"{f.parent.name} {a}: add us/row = {fa:.2f} + {fb:.3f} lnN "
        f"(fit {FIT_LO:.0e}..{FIT_HI:.0e})"
    )
    for o in g:
        if o["n"] > FIT_HI:
            pred = fa + fb * np.log(o["n"])
            out.append(
                f"  N={o['n']:.0e} meas {o['add_us_row']:.1f} fit {pred:.1f}"
                f" ratio {o['add_us_row'] / pred:.2f}"
                f" disk {o['disk_gib']:.1f} GiB"
            )


def cap_run(ax, axr, f, a, col, lab, out):
    r = rows(f)
    g = grid(r)
    if not g:
        return
    p = [o for o in r if o.get("progress")]
    n = [o["n"] for o in g]
    rd = lambda o: max(o["rd_b_row"], 0.1)
    scatter(ax, p, lambda o: o["add_us_row"], col)
    ax.plot(n, [o["add_us_row"] for o in g], "o-", color=col, lw=0.9, ms=2.5, label=lab)
    scatter(axr, p, rd, col)
    axr.plot(n, [rd(o) for o in g], "o-", color=col, lw=0.9, ms=2.5, label=lab)
    fit_lines(ax, f, a, col, g, out)


def cap_panel(ax, axr, d, out):
    for a, col in C.items():
        for i, f in enumerate(sorted(d.glob(f"cap1G/r*/bpa_D100_1G_pb{a}.jsonl"))):
            cap_run(ax, axr, f, a, col, LAB[a] if i == 0 else None, out)
    style(ax, "(a) add time, 1 GB cap, D=100", "us per added row")
    ax.set_yscale("log")
    ax.set_ylim(1, 3e3)
    style(axr, "(b) disk reads during adds", "bytes read per added row")
    axr.set_yscale("log")
    axr.set_ylim(0.08, 3e4)
    axr.set_xlim(3e4, 4e7)
    for x in (ax, axr):
        x.axvline(2.3e5, color="#8c8c8c", lw=0.5, ls=":")
        x.axvline(2.5e7, color="#8c8c8c", lw=0.5, ls="--")
        x.legend(fontsize=5, frameon=False, loc="upper left")


def cold_panel(ax, d, key, title, ylab, scale, out):
    for a, col in C.items():
        r = rows(d / f"cold/cold_D100_1G_pb{a}.jsonl")
        if not r:
            continue
        n = np.array([o["n"] for o in r], float)
        med = np.array([np.median(o[key]) for o in r]) * scale
        q1 = np.array([np.percentile(o[key], 25) for o in r]) * scale
        q3 = np.array([np.percentile(o[key], 75) for o in r]) * scale
        ax.fill_between(n, q1, q3, color=col, alpha=0.15, lw=0)
        ax.plot(n, med, "o-", color=col, lw=0.9, ms=2.5, label=LAB[a])
        m = n >= 1e2
        if m.sum() >= 3:
            b, a0 = np.polyfit(np.log(n[m]), med[m], 1)
            out.append(f"cold {key} {a}: {a0:.3f} + {b:.4f} lnN (N>=1e2)")
            for nn, yy in zip(n, med):
                out.append(f"  N={nn:.0e} median {yy:.3f}")
    style(ax, title, ylab)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=5, frameon=False, loc="upper left")


def main():
    d = Path(sys.argv[1])
    out = []
    fig, ax = plt.subplots(2, 2, figsize=(7.0, 4.6))
    cap_panel(ax[0, 0], ax[0, 1], d, out)
    cold_panel(
        ax[1, 0],
        d,
        "cold_add_us_row",
        "(c) cold add, store paged out",
        "ms per row",
        1e-3,
        out,
    )
    cold_panel(
        ax[1, 1],
        d,
        "cold_q_us",
        "(d) cold query, store paged out",
        "ms per query",
        1e-3,
        out,
    )
    fig.tight_layout()
    fig.savefig(d / "fig_0246.pdf")
    if len(sys.argv) > 2:
        fig.savefig(sys.argv[2], dpi=90)
    (d / "fit_0246.txt").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
