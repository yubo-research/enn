import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

B, O, GR = "#2a78d6", "#eb6834", "#8a8984"
D = Path(sys.argv[1])
OUT = Path(sys.argv[2])
ARMS = (
    ("cap256", "256M", "3000aph", O, "-", "256 MB, buffers R=3000"),
    ("cap256", "256M", "1000aph", "#c9a227", "-", "256 MB, buffers R=1000"),
    ("cap256", "256M", "0", B, "-", "256 MB, no buffers (default)"),
    ("cap1G", "1G", "3000aph", O, "--", "1 GB, buffers R=3000"),
    ("cap1G", "1G", "0", B, "--", "1 GB, no buffers (default)"),
    ("cap1G", "1G", "0_main", GR, ":", "1 GB, main 3246c4e"),
)
FIT_LO, FIT_HI = 1e2, 3e4


def jl(p):
    return (
        [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
        if Path(p).exists()
        else []
    )


def style(ax, title, ylab):
    ax.set(xscale="log", yscale="log", xlabel="N (rows)", ylabel=ylab)
    ax.set_title(title, fontsize=8, loc="left")
    ax.xaxis.label.set_size(7)
    ax.yaxis.label.set_size(7)
    ax.tick_params(labelsize=6)
    ax.grid(True, color="#e4e3df", lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def series(sub, cap, arm):
    rows = [o for o in jl(D / sub / f"capq_D100_{cap}_pb{arm}.jsonl") if o["n"] >= 10]
    return rows


def fit(rows, key):
    pts = [(np.log(o["n"]), o[key]) for o in rows if FIT_LO <= o["n"] <= FIT_HI]
    if len(pts) < 3:
        return None
    x, y = np.array(pts).T
    return np.polyfit(x, y, 1)


def bar_style(a, title, ylab):
    a.set_title(title, fontsize=8, loc="left")
    a.set_ylabel(ylab, fontsize=7)
    a.tick_params(labelsize=6)
    a.grid(True, axis="y", color="#e4e3df", lw=0.5)
    a.spines[["top", "right"]].set_visible(False)


def panel_e(a):
    arms = (
        ("0", "no buffers", B),
        ("3000", "R=3000", O),
        ("3000c", "R=3000 charged", "#c9a227"),
    )
    for j, (arm, lb, c) in enumerate(arms):
        v = []
        for N in ("1e6", "3e6"):
            p = D / "attr" / f"qattr_{arm}_{N}.json"
            v.append(
                json.loads(p.read_text())["runs"][0]["scanned_rows_per_q"]
                if p.exists()
                else np.nan
            )
        a.bar(np.arange(2) + (j - 1) * 0.27, v, 0.27, color=c, label=lb)
    a.set_yscale("log")
    a.set_xticks([0, 1], ["N=1e6", "N=3e6"])
    bar_style(
        a, "(e) Rows scanned per query, 1 GB (qattr)", "rows / query (leaf + buffer)"
    )
    a.legend(fontsize=5.5, frameon=False)


def panel_f(a, lines):
    Ns = ("1e5", "1e6", "3e6")
    for j, (v, lb, c) in enumerate(
        (("base", "pool k+30 (default)", B), ("extra0", "pool k", O))
    ):
        kb, ms = [], []
        for N in Ns:
            p = D / "extra" / f"extra_{N}.json"
            runs = (
                [r for r in json.loads(p.read_text())["runs"] if r["variant"] == v]
                if p.exists()
                else []
            )
            kb.append(np.mean([r["q_rd_b"] for r in runs]) / 1e3 if runs else np.nan)
            ms.append(np.mean([r["q_us_mean"] for r in runs]) / 1e3 if runs else np.nan)
        x = np.arange(len(Ns)) + (j - 0.5) * 0.38
        a.bar(x, kb, 0.38, color=c, label=lb)
        for xi, k, m in zip(x, kb, ms):
            a.text(xi, k, f"{m:.1f} ms", ha="center", va="bottom", fontsize=5)
        lines.append(
            f"exact pool {lb}: "
            + "; ".join(f"N={N} {k:.0f} KB {m:.1f} ms" for N, k, m in zip(Ns, kb, ms))
        )
    a.set_xticks(range(len(Ns)), [f"N={N}" for N in Ns])
    bar_style(
        a,
        "(f) Read/query vs exact re-read pool (1 GB)",
        "KB read / query (label: ms/query)",
    )
    a.legend(fontsize=5.5, frameon=False)


def summary_lines(lb, r, n, q, rd, ad):
    out = []
    p = fit(r, "q_us_mean")
    if p is not None:
        pred = (p[0] * np.log(n) + p[1]) / 1e3
        out.append(
            f"{lb}: query fit {FIT_LO:.0e}-{FIT_HI:.0e}: {p[1]:.0f} + {p[0]:.0f} lnN us; "
            + "; ".join(
                f"N={k:.0e} meas {a:.1f} ms fit {b:.2f} ms ratio {a / b:.1f}"
                for k, a, b in zip(n, q, pred)
                if k >= 1e5
            )
        )
    out.append(
        f"{lb}: "
        + "; ".join(
            f"N={k:.0e} q {a:.2f} ms rd {b:.3f} MB add {d:.1f} us"
            for k, a, b, d in zip(n, q, rd, ad)
            if k >= 1e3
        )
    )
    return out


def plot_arm(ax, arm, lines):
    sub, cap, name, c, ls, lb = arm
    r = series(sub, cap, name)
    if not r:
        return
    n = np.array([o["n"] for o in r])
    q = np.array([o["q_us_mean"] for o in r]) / 1e3
    qse = np.array([o["q_us_se"] for o in r]) / 1e3
    rd = np.array([o["q_rd_b"] for o in r]) / 1e6
    ad = np.array([o["add_us_row"] for o in r])
    fd = np.array([o["fd_us_mean"] for o in r]) / 1e3
    kw = {"ls": ls, "color": c, "ms": 3, "lw": 1.2, "label": lb}
    mk = "o" if cap == "256M" else "s"
    ax[0, 0].errorbar(n, q, yerr=qse, fmt=mk, **kw)
    ax[0, 1].plot(n, np.maximum(rd, 1e-3), mk, **kw)
    m = n >= 1e3
    ax[1, 0].plot(n[m], ad[m], mk, **kw)
    ax[1, 1].plot(n, fd, mk, **kw)
    lines.extend(summary_lines(lb, r, n, q, rd, ad))


def main():
    fig, ax = plt.subplots(3, 2, figsize=(7.0, 7.4))
    lines = []
    for arm in ARMS:
        plot_arm(ax, arm, lines)
    style(
        ax[0, 0],
        "(a) Posterior query time under the cap",
        "ms / query (mean $\\pm$ SE)",
    )
    style(ax[0, 1], "(b) Device bytes read per query", "MB / query (floor 1e-3)")
    style(ax[1, 0], "(c) Add time under the cap", "us / row (1000-row adds)")
    style(
        ax[1, 1],
        "(d) posterior_function_draw time under the cap",
        "ms / call (1 point, 1 draw)",
    )
    ax[0, 0].legend(fontsize=5.5, frameon=False)
    panel_e(ax[2, 0])
    panel_f(ax[2, 1], lines)
    fig.tight_layout()
    fig.savefig(OUT)
    fig.savefig(OUT.with_suffix(".png"), dpi=150)
    (OUT.parent / "fit_0250.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


main()
