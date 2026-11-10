#!/usr/bin/env python
"""Summarize ops/scaling.py JSON-lines output: figures and LaTeX tables."""

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
WINDOWS = ((1e3, 1e5), (1e5, 1e7), (1e3, 1e7))


def load(paths: list[Path]) -> dict[tuple[int, int], list[dict]]:
    out: dict[tuple[int, int], list[dict]] = {}
    for p in paths:
        rows = [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
        if rows:
            out[(rows[0]["num_dim"], rows[0]["seed"])] = rows
    return out


def series(rows: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    pts = [(r["n_actual"], r[key]) for r in rows if r[key] is not None]
    return np.array([p[0] for p in pts], float), np.array([p[1] for p in pts], float)


def slope(n: np.ndarray, t: np.ndarray, lo: float, hi: float) -> float | None:
    m = (n >= lo * 0.99) & (n <= hi * 1.05)
    if m.sum() < 3:
        return None
    return float(np.polyfit(np.log(n[m]), np.log(t[m]), 1)[0])


def fmt_t(x: float) -> str:
    return f"{x:.2e}".replace("e-0", "e-").replace("e+0", "e")


@click.command()
@click.argument("data_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("out_dir", type=click.Path(file_okay=False))
def main(data_dir: str, out_dir: str) -> None:
    runs = load(sorted(Path(data_dir).glob("*.jsonl")))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d10 = {k: v for k, v in runs.items() if k[0] == 10}
    # Figure: 2x2 log-log, seeds as thin lines, O(N) reference.
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    for ax, (key, label) in zip(axes.ravel(), METRICS.items(), strict=True):
        for (d, s), rows in sorted(runs.items()):
            n, t = series(rows, key)
            kw = {"lw": 1.6} if d == 10 else {"lw": 1.0, "ls": "--"}
            ax.loglog(n, t, label=f"d={d}" + (f", seed {s}" if d == 10 else ""), **kw)
        n, t = series(next(iter(d10.values())), key)
        i = int(np.argmin(np.abs(n - 1e4)))
        ax.loglog(n, t[i] * n / n[i], "k:", lw=1, label=r"$\propto N$")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("N", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, which="both")
    axes[0, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "scaling.pdf")
    # Figure: local slope between adjacent grid points (d=10, median over seeds).
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for key, label in METRICS.items():
        sl = []
        for rows in d10.values():
            n, t = series(rows, key)
            m = n >= 300
            n, t = n[m], t[m]
            sl.append(np.diff(np.log(t)) / np.diff(np.log(n)))
        k = min(len(s) for s in sl)
        sl = np.array([s[:k] for s in sl])
        n, _ = series(next(iter(d10.values())), key)
        n = n[n >= 300][1 : k + 1]
        ax.semilogx(n, np.median(sl, axis=0), marker="o", ms=3, lw=1, label=label)
    ax.axhline(1, color="k", ls=":", lw=1)
    ax.set_xlabel("N", fontsize=8)
    ax.set_ylabel("local log-log slope", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=5)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "slopes.pdf")

    # Tables.
    lines = []
    ref = sorted(d10.items())
    ns = sorted({int(r["n"]) for _, rows in ref for r in rows})
    for n0 in ns:
        cells = [f"{n0:,}".replace(",", r"\,")]
        for key in METRICS:
            vals = []
            for _, rows in ref:
                for r in rows:
                    if r["n"] == n0 and r[key] is not None:
                        vals.append(r[key])
            cells.append(fmt_t(float(np.median(vals))) if vals else "--")
        lines.append(" & ".join(cells) + r" \\")
    (out / "times.tex").write_text("\n".join(lines).removesuffix(r" \\") + "%\n")

    lines = []
    for key, label in METRICS.items():
        cells = [label]
        for lo, hi in WINDOWS:
            sl = [
                s
                for rows in d10.values()
                if (s := slope(*series(rows, key), lo, hi)) is not None
            ]
            if sl:
                cells.append(
                    f"{np.mean(sl):.2f}"
                    + (f" $\\pm$ {np.std(sl, ddof=1):.2f}" if len(sl) > 1 else "")
                )
            else:
                cells.append("--")
        lines.append(" & ".join(cells) + r" \\")
    (out / "slopes.tex").write_text("\n".join(lines).removesuffix(r" \\") + "%\n")

    lines = []
    for (d, s), rows in sorted(runs.items()):
        if d == 10:
            continue
        cells = [str(d)]
        for key in METRICS:
            for lo, hi in ((1e3, 1e5), (1e5, 3e6)):
                v = slope(*series(rows, key), lo, hi)
                cells.append("--" if v is None else f"{v:.2f}")
        lines.append(" & ".join(cells) + r" \\")
    for (d, s), rows in sorted(runs.items()):
        if d == 10 and s == 0:
            cells = ["10"]
            for key in METRICS:
                for lo, hi in ((1e3, 1e5), (1e5, 3e6)):
                    v = slope(*series(rows, key), lo, hi)
                    cells.append("--" if v is None else f"{v:.2f}")
            lines.append(" & ".join(cells) + r" \\")
    (out / "dims.tex").write_text("\n".join(lines).removesuffix(r" \\") + "%\n")

    # Ratio test: time(N2)/time(N1) versus N2/N1.
    lines = []
    for key, label in METRICS.items():
        cells = [label]
        for n1, n2 in ((1e3, 1e5), (1e5, 1e7), (1e6, 1e7)):
            rs = []
            for rows in d10.values():
                n, t = series(rows, key)
                a, b = np.argmin(np.abs(n - n1)), np.argmin(np.abs(n - n2))
                if abs(n[b] - n2) / n2 < 0.05:
                    rs.append(t[b] / t[a])
            cells.append(f"{np.median(rs):.1f}" if rs else "--")
        lines.append(" & ".join(cells) + r" \\")
    (out / "ratios.tex").write_text("\n".join(lines).removesuffix(r" \\") + "%\n")
    print("seeds (d=10):", sorted(s for _, s in d10), "runs:", sorted(runs))


if __name__ == "__main__":
    main()
