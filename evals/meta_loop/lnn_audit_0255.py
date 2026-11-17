import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPORTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "reports"
)
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPORTS, "report-0255-data")
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)
GRID = "#d9d8d4"
INK = "#52514e"

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.edgecolor": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.6,
        "lines.markersize": 4,
        "figure.dpi": 150,
    }
)


def load(rel):
    with open(os.path.join(REPORTS, rel)) as f:
        return [json.loads(line) for line in f if line.strip()]


def series(rows, key, nmin=1):
    pts = [
        (r["n"], r[key])
        for r in rows
        if r.get("n", 0) >= nmin and r.get(key) is not None and r[key] > 0
    ]
    pts.sort()
    return np.array([p[0] for p in pts], float), np.array([p[1] for p in pts], float)


def ln_fit(n, t, lo, hi):
    m = (n >= lo) & (n <= hi)
    if m.sum() < 2:
        return None
    b, a = np.polyfit(np.log(n[m]), t[m], 1)
    return a, b


def local_exponent(n, t):
    ln_n, ln_t = np.log(n), np.log(t)
    return n[1:], np.diff(ln_t) / np.diff(ln_n)


def ln_exponent_reference(a, b, n):
    return b / (a + b * np.log(n))


ARMS = {
    "1G_D100": [
        (
            "report-0250-data/cap1G/capq_D100_1G_pb0.jsonl",
            "tree, no buffers",
            BLUE,
            "-",
        ),
        (
            "report-0250-data/cap1G/capq_D100_1G_pb3000aph.jsonl",
            "tree, every-level buffers R=3000",
            ORANGE,
            "-",
        ),
        (
            "report-0250-data/cap1G/capq_D100_1G_pb0_main.jsonl",
            "main (RAM-resident rows)",
            INK,
            "--",
        ),
    ],
    "256M_D100": [
        (
            "report-0250-data/cap256/capq_D100_256M_pb0.jsonl",
            "tree, no buffers",
            BLUE,
            "-",
        ),
        (
            "report-0250-data/cap256/capq_D100_256M_pb3000aph.jsonl",
            "tree, every-level buffers R=3000",
            ORANGE,
            "-",
        ),
        (
            "report-0254-data/capq100/capq_D100_256M_pb0_c256_r1.jsonl",
            "tree, 256-row leaves",
            AQUA,
            "-",
        ),
    ],
    "256M_D10": [
        (
            "report-0254-data/capq/capq_D10_256M_pb0_c64_r1.jsonl",
            "tree, no buffers (64-row leaves)",
            BLUE,
            "-",
        ),
        (
            "report-0254-data/capq/capq_D10_256M_pb0_c256_big.jsonl",
            "tree, 256-row leaves",
            AQUA,
            "-",
        ),
        (
            "report-0252-data/cap256/capq_D10_256M_pb0_r16a.jsonl",
            "tree, split-time reassignment",
            VIOLET,
            "-",
        ),
    ],
}
PANEL_TITLES = {
    "1G_D100": "cap 1 GB, D = 100 (0250)",
    "256M_D100": "cap 256 MB, D = 100 (0250, 0254)",
    "256M_D10": "cap 256 MB, D = 10 (0252, 0254)",
}
CAP_BYTES = {"1G_D100": 2**30, "256M_D100": 2**28, "256M_D10": 2**28}
RAW_ROW_BYTES = {"1G_D100": 808, "256M_D100": 808, "256M_D10": 88}


def draw_fit(ax, fit, lo, hi, floor, color):
    if fit is None:
        return
    xs = np.geomspace(lo, hi, 50)
    ax.plot(
        xs,
        np.maximum(fit[0] + fit[1] * np.log(xs), floor),
        color=color,
        lw=1.0,
        alpha=0.6,
    )


def max_step(t):
    return float(np.max(t[1:] / t[:-1])) if len(t) > 1 else float("nan")


def meas_over_fit(n, t, fit):
    return float(t[-1] / (fit[0] + fit[1] * math.log(n[-1]))) if fit else float("nan")


def plot_capped_arm(ax_add, ax_q, key, arm):
    rel, label, color, ls = arm
    rows = load(rel)
    n, add = series(rows, "add_us_row", nmin=1000)
    nq, q = series(rows, "q_us_med", nmin=100)
    q = q / 1000.0
    ax_add.plot(n, add, ls, color=color, marker="o", label=label)
    ax_q.plot(nq, q, ls, color=color, marker="o", label=label)
    fit_add = ln_fit(n, add, 1e3, 1e5)
    fit_q = ln_fit(nq, q, 1e4, 1e5)
    draw_fit(ax_add, fit_add, 1e3, n.max(), 0.05, color)
    draw_fit(ax_q, fit_q, 1e4, nq.max(), 0.005, color)
    return {
        "panel": key,
        "arm": label,
        "n_max": int(n.max()),
        "add_last_us": float(add[-1]),
        "add_max_step": max_step(add),
        "add_meas_over_fit": meas_over_fit(n, add, fit_add),
        "q_last_ms": float(q[-1]),
        "q_max_step": max_step(q),
        "q_meas_over_fit": meas_over_fit(nq, q, fit_q),
    }


def style_capped_panel(ax_add, ax_q, key, col):
    ten_x = 10 * CAP_BYTES[key] / RAW_ROW_BYTES[key]
    for ax in (ax_add, ax_q):
        ax.axvline(ten_x, color=GRID, lw=1.2, ls=":", zorder=0)
        ax.set_yscale("log")
    ax_q.text(
        ten_x, 0.025, "10×C\n(raw x)", ha="center", va="bottom", fontsize=6, color=INK
    )
    ax_add.set_xscale("log")
    ax_add.set_title(PANEL_TITLES[key])
    ax_add.set_ylim(1, 5000)
    ax_q.set_ylim(0.02, 5000)
    ax_q.set_xlabel("N (rows)")
    if col == 0:
        ax_add.set_ylabel("add, µs per row\n(lower is better)")
        ax_q.set_ylabel("posterior, ms per query\n(lower is better)")
    ax_add.legend(loc="upper left", frameon=False, handlelength=1.6)


def fig_capped():
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), sharex=True)
    summary = []
    for col, key in enumerate(["1G_D100", "256M_D100", "256M_D10"]):
        ax_add, ax_q = axes[0, col], axes[1, col]
        summary.extend(plot_capped_arm(ax_add, ax_q, key, arm) for arm in ARMS[key])
        style_capped_panel(ax_add, ax_q, key, col)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_0255_capped.pdf"))
    fig.savefig(os.path.join(OUT, "fig_0255_capped.png"))
    return summary


def plot_exponent_arm(axes, rows, label, color):
    n, add = series(rows, "add_us_row", nmin=1000)
    nq, q = series(rows, "q_us_med", nmin=1000)
    xa, ea = local_exponent(n, add)
    xq, eq = local_exponent(nq, q)
    axes[0].plot(xa, ea, "-o", color=color, label=label)
    axes[1].plot(xq, eq, "-o", color=color, label=label)


def fig_exponent():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
    picks = [
        ("1G_D100", 0),
        ("1G_D100", 1),
        ("256M_D100", 0),
        ("256M_D10", 0),
        ("256M_D10", 1),
    ]
    colors = [BLUE, ORANGE, AQUA, VIOLET, MAGENTA]
    for (key, idx), color in zip(picks, colors):
        rel, label, _, _ = ARMS[key][idx]
        tag = PANEL_TITLES[key].split(" (")[0]
        plot_exponent_arm(axes, load(rel), f"{label}; {tag}", color)
    ref = np.geomspace(1e3, 3e7, 50)
    for ax, title in zip(axes, ["add", "posterior query"]):
        ax.plot(
            ref,
            ln_exponent_reference(0.0, 1.0, ref),
            color=INK,
            lw=1,
            ls="--",
            label="a + b ln N with a = 0 (upper bound for any ln N law)",
        )
        ax.axhline(1.0, color=GRID, lw=1)
        ax.text(1.2e3, 1.03, "linear in N", fontsize=6, color=INK)
        ax.set_xscale("log")
        ax.set_xlabel("N (rows)")
        ax.set_title(f"local exponent of {title} cost")
    axes[0].set_ylabel(
        "d ln t / d ln N between checkpoints\n(lower is better; ln N law ≤ dashed line)"
    )
    axes[0].set_ylim(-0.3, 3.2)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=6.5)
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    fig.savefig(os.path.join(OUT, "fig_0255_exponent.pdf"))
    fig.savefig(os.path.join(OUT, "fig_0255_exponent.png"))


def median_and_se(rows, key, scale):
    med = np.array([np.median(r[key]) / scale for r in rows])
    se = np.array([np.std(r[key]) / math.sqrt(len(r[key])) / scale for r in rows])
    return med, se


def draw_cold_panel(ax, n, y, se, color, label, name):
    ax.errorbar(n, y, yerr=se, fmt="o", color=color, label=label, capsize=2, lw=1)
    fit = ln_fit(n, y, 1e4, 1e6)
    if fit is None:
        return None
    xs = np.geomspace(1e4, 1e6, 20)
    ax.plot(xs, fit[0] + fit[1] * np.log(xs), color=color, lw=1)
    return (label, name, fit[0], fit[1])


def fig_cold():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    out = []
    for rel, label, color in [
        ("report-0235-data/cold_D10_1G.jsonl", "D = 10", BLUE),
        ("report-0235-data/cold_D100_1G.jsonl", "D = 100", ORANGE),
    ]:
        rows = load(rel)
        n = np.array([r["n"] for r in rows], float)
        q, q_se = median_and_se(rows, "cold_q_us", 1000.0)
        a, a_se = median_and_se(rows, "cold_add_us_row", 1000.0)
        out.append(draw_cold_panel(axes[0], n, q, q_se, color, label, "cold query ms"))
        out.append(
            draw_cold_panel(axes[1], n, a, a_se, color, label, "cold add ms/row")
        )
    axes[0].set_ylabel("ms per posterior, store evicted first\n(lower is better)")
    axes[1].set_ylabel("ms per added row, store evicted first\n(lower is better)")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("N (rows)")
        ax.legend(frameon=False, loc="upper left")
    axes[0].set_title("true on-disk query cost (0235); lines: a + b ln N")
    axes[1].set_title("true on-disk add cost (0235)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_0255_cold.pdf"))
    fig.savefig(os.path.join(OUT, "fig_0255_cold.png"))
    return [c for c in out if c is not None]


def onset(rows, threshold=100.0):
    n, add = series(rows, "add_us_row", nmin=1000)
    for i in range(1, len(n)):
        if add[i] >= threshold and add[i - 1] < threshold:
            f = (math.log(threshold) - math.log(add[i - 1])) / (
                math.log(add[i]) - math.log(add[i - 1])
            )
            return math.exp(
                math.log(n[i - 1]) + f * (math.log(n[i]) - math.log(n[i - 1]))
            )
    return None


def fig_onset():
    pts = [
        (
            "report-0218-data/bpa_D100_64M.jsonl",
            2**26,
            190,
            "0218, 64 MB",
            BLUE,
            (6, -9),
        ),
        (
            "report-0218-data/bpa_D100_256M.jsonl",
            2**28,
            190,
            "0218, 256 MB",
            BLUE,
            (-6, -11),
        ),
        (
            "report-0218-data/bpa_D100_1G.jsonl",
            2**30,
            190,
            "0218, 1 GB",
            BLUE,
            (6, -10),
        ),
        (
            "report-0241-data/cap256/bpa_D100_256M_c0.jsonl",
            2**28,
            190,
            "0241, 8-bit",
            ORANGE,
            (-46, 2),
        ),
        (
            "report-0241-data/cap256/bpa_D100_256M_c2i.jsonl",
            2**28,
            140,
            "0241, 4-bit residual",
            ORANGE,
            (6, -3),
        ),
        (
            "report-0250-data/cap256/capq_D100_256M_pb0.jsonl",
            2**28,
            190,
            "0250 (coarse grid)",
            AQUA,
            (-72, 5),
        ),
    ]
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    rows_out = []
    for rel, cap, bytes_row, label, color, off in pts:
        n0 = onset(load(rel))
        if n0 is None:
            continue
        x = cap / bytes_row
        ax.plot(x, n0, "o", color=color)
        ax.annotate(
            label,
            (x, n0),
            textcoords="offset points",
            xytext=off,
            fontsize=6,
            color=INK,
        )
        rows_out.append((label, cap, bytes_row, n0, n0 * bytes_row / cap))
    xs = np.geomspace(2e5, 1e7, 10)
    ax.fill_between(
        xs,
        0.5 * xs,
        xs,
        color=GRID,
        alpha=0.5,
        lw=0,
        label="0.5–1 × C / (hot bytes per row)",
    )
    ax.plot(xs, xs, color=INK, lw=1, ls="--", label="onset = C / (hot bytes per row)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("C / hot bytes per row (rows)")
    ax.set_ylabel("N at add-cliff onset (first ≥ 100 µs/row)")
    ax.set_title("the add cliff sits where the hot set fills the cap", fontsize=7.5)
    ax.legend(frameon=False, loc="upper left", fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_0255_onset.pdf"))
    fig.savefig(os.path.join(OUT, "fig_0255_onset.png"))
    return rows_out


def main():
    summary = fig_capped()
    fig_exponent()
    cold = fig_cold()
    onsets = fig_onset()
    with open(os.path.join(OUT, "summary.txt"), "w") as f:
        f.write(
            "panel | arm | n_max | add_last_us | add_max_step | add_meas/fit | q_last_ms | q_max_step | q_meas/fit\n"
        )
        f.writelines(
            f"{s['panel']} | {s['arm']} | {s['n_max']} | {s['add_last_us']:.1f} | {s['add_max_step']:.2f} | "
            f"{s['add_meas_over_fit']:.2f} | {s['q_last_ms']:.2f} | {s['q_max_step']:.2f} | {s['q_meas_over_fit']:.2f}\n"
            for s in summary
        )
        f.write("\ncold fits (0235), N in [1e4,1e6]: label | metric | a | b per lnN\n")
        f.writelines(f"{c[0]} | {c[1]} | {c[2]:.3f} | {c[3]:.3f}\n" for c in cold)
        f.write("\nadd-cliff onsets: label | cap | hot B/row | onset N | onset*B/C\n")
        f.writelines(
            f"{o[0]} | {o[1]} | {o[2]} | {o[3]:.3g} | {o[4]:.2f}\n" for o in onsets
        )
    with open(os.path.join(OUT, "summary.txt")) as f:
        print(f.read())


if __name__ == "__main__":
    main()
