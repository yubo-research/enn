#!/usr/bin/env python
"""Summarize ops/scaling.py output for the d=1000, 10-metric study.

Writes figures and LaTeX table bodies. The d=10 one-metric data from the first
scaling study is overlaid for comparison (time only, seed 0).
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METRICS = {
    "add_row_s": "Add, batched (s/row)",
    "add1_s": "Add, single row (s/call)",
    "query_batch_s": "Query, 1000 points (s/call)",
    "query1_s": "Query, 1 point (s/call)",
}
WINDOWS = ((1e3, 1e4), (1e4, 1e5), (1e3, 1e5))


def load(paths: list[Path]) -> dict[int, list[dict]]:
    out = {}
    for p in paths:
        rows = [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
        if rows:
            out[rows[0]["seed"]] = rows
    return out


def series(rows: list[dict], key: str, sub=None):
    pts = [
        (r["n_actual"], sub(r) if sub else r[key])
        for r in rows
        if (sub or r[key] is not None)
    ]
    return np.array([p[0] for p in pts], float), np.array([p[1] for p in pts], float)


def slope(n, t, lo, hi):
    m = (n >= lo * 0.99) & (n <= hi * 1.05)
    if m.sum() < 3:
        return None
    return float(np.polyfit(np.log(n[m]), np.log(t[m]), 1)[0])


def fmt_t(x: float) -> str:
    return f"{x:.2e}".replace("e-0", "e-").replace("e+0", "e")


def rows_of(vals) -> str:
    return "\n".join(vals).removesuffix(r" \\") + "%\n"


@click.command()
@click.argument("data_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("base_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("out_dir", type=click.Path(file_okay=False))
def main(data_dir: str, base_dir: str, out_dir: str) -> None:
    runs = load(sorted(Path(data_dir).glob("*.jsonl")))
    base = load([Path(base_dir) / "d10_s0.jsonl"])[0]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    anon = lambda r: r["rss_mib"]["RssAnon"]
    # Time figure.
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    for ax, (key, label) in zip(axes.ravel(), METRICS.items(), strict=True):
        for s, rows in sorted(runs.items()):
            n, t = series(rows, key)
            ax.loglog(n, t, lw=1.4, label=f"d=1000, 10 metrics, seed {s}")
        n, t = series(base, key)
        m = n <= 1e5
        ax.loglog(n[m], t[m], "--", lw=1, color="gray", label="d=10, 1 metric")
        n, t = series(next(iter(runs.values())), key)
        i = int(np.argmin(np.abs(n - 1e4)))
        ax.loglog(n, t[i] * n / n[i], "k:", lw=1, label=r"$\propto N$")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("N", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, which="both")
    axes[0, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "scaling.pdf")
    # Memory figure: anonymous resident memory and per-row cost.
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.6))
    raw = lambda n: n * (1000 + 10) * 8 / 2**20
    for s, rows in sorted(runs.items()):
        n, m = series(rows, "", anon)
        a1.loglog(n, m, lw=1.4, label=f"seed {s}")
        k = n >= 1000
        a2.semilogx(n[k], m[k] * 2**20 / n[k], lw=1.4, label=f"seed {s}")
    n, _ = series(next(iter(runs.values())), "", anon)
    a1.loglog(n, raw(n), "k:", lw=1, label="raw float64 data")
    a2.axhline(1010 * 8, color="k", ls=":", lw=1, label="raw float64 row")
    a1.set_ylabel("anonymous resident memory (MiB)", fontsize=8)
    a2.set_ylabel("bytes per stored row", fontsize=8)
    for ax in (a1, a2):
        ax.set_xlabel("N", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "memory.pdf")
    # Tables.
    ns = sorted({int(r["n"]) for rows in runs.values() for r in rows})
    lines = []
    for n0 in ns:
        cells = [f"{n0:,}".replace(",", r"\,")]
        for key in METRICS:
            v = [
                r[key]
                for rows in runs.values()
                for r in rows
                if r["n"] == n0 and r[key] is not None
            ]
            cells.append(fmt_t(float(np.median(v))) if v else "--")
        mem = [anon(r) for rows in runs.values() for r in rows if r["n"] == n0]
        pk = [r["max_rss_mib"] for rows in runs.values() for r in rows if r["n"] == n0]
        cells.append(f"{np.median(mem):.0f}")
        cells.append(f"{np.median(pk):.0f}")
        lines.append(" & ".join(cells) + r" \\")
    (out / "times.tex").write_text(rows_of(lines))
    lines = []
    labels = ("Add b.", "Add 1", "Q 1000", "Q 1", "Memory")
    for key, label in zip((*METRICS, "mem"), labels, strict=True):
        cells = [label]
        for lo, hi in WINDOWS:
            sl = []
            for rows in runs.values():
                n, t = series(rows, key, anon if key == "mem" else None)
                if (v := slope(n, t, lo, hi)) is not None:
                    sl.append(v)
            if sl:
                sd = f" $\\pm$ {np.std(sl, ddof=1):.2f}" if len(sl) > 1 else ""
                cells.append(f"{np.mean(sl):.2f}{sd}")
            else:
                cells.append("--")
        lines.append(" & ".join(cells) + r" \\")
    (out / "slopes.tex").write_text(rows_of(lines))
    lines = []
    short = dict(zip(METRICS, ("Add b.", "Add 1", "Q 1000", "Q 1"), strict=True))
    for key in METRICS:
        cells = [short[key]]
        for n1, n2 in ((1e3, 1e4), (1e4, 1e5), (1e3, 1e5)):
            rs, rb = [], []
            for rows in runs.values():
                n, t = series(rows, key)
                a, b = np.argmin(np.abs(n - n1)), np.argmin(np.abs(n - n2))
                if abs(n[b] - n2) / n2 < 0.05:
                    rs.append(t[b] / t[a])
            n, t = series(base, key)
            a, b = np.argmin(np.abs(n - n1)), np.argmin(np.abs(n - n2))
            rb.append(t[b] / t[a])
            cells.append(f"{np.median(rs):.1f}" if rs else "--")
            cells.append(f"{rb[0]:.1f}")
        lines.append(" & ".join(cells) + r" \\")
    (out / "ratios.tex").write_text(rows_of(lines))
    print("seeds:", sorted(runs))


if __name__ == "__main__":
    main()
