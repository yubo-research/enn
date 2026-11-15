import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

KEYS = ["add_us_row", "query_s", "maxrss_mb", "nrmse", "loglik"]
LABELS = ["add µs/row", "query s (1000 pts)", "peak RSS MB", "NRMSE", "loglik"]


def read_runs(path):
    lines = path.read_text().splitlines()
    recs = [json.loads(ln) for ln in lines if ln.startswith("{")]
    return {(r["D"], r["n"], r["seed"]): r for r in recs}


def paired_stats(h, x):
    df = h - x
    se = df.std(ddof=1) / np.sqrt(len(df)) if len(df) > 1 else float("nan")
    p = stats.ttest_1samp(df, 0).pvalue if len(df) > 1 and df.std() > 0 else 1.0
    return (h.mean(), x.mean(), df.mean(), se, p)


def build_rows(head, main):
    rows = []
    for dim in (10, 100):
        for n in (1000, 10000, 100000, 1000000):
            ks = [k for k in head if k[:2] == (dim, n) and k in main]
            if not ks:
                continue
            out = {"D": dim, "n": n, "pairs": len(ks)}
            for m in KEYS:
                h = np.array([head[k][m] for k in ks])
                x = np.array([main[k][m] for k in ks])
                out[m] = paired_stats(h, x)
            rows.append(out)
    return rows


def write_table(rows, path):
    with open(path, "w") as f:
        for r in rows:
            f.write(f"D={r['D']} N={r['n']:.0e} pairs={r['pairs']}\n")
            for m in KEYS:
                h, x, df, se, p = r[m]
                f.write(
                    f"  {m:11s} head={h:10.4g} main={x:10.4g} diff={df:+.3g} "
                    f"(SE {se:.2g}, p={p:.3f}) ratio={h / x:.2f}\n"
                )


def plot_rows(rows, path):
    fig, ax = plt.subplots(1, 5, figsize=(16, 3.2))
    for a, m, lab in zip(ax, KEYS, LABELS):
        for dim, c in ((10, "#1f77b4"), (100, "#d62728")):
            rr = [r for r in rows if r["D"] == dim]
            n = [r["n"] for r in rr]
            a.plot(n, [r[m][0] for r in rr], "o-", c=c, label=f"head D={dim}")
            a.plot(
                n, [r[m][1] for r in rr], "s--", c=c, mfc="none", label=f"main D={dim}"
            )
        a.set_xscale("log")
        a.set_title(lab, fontsize=9)
        a.set_xlabel("N")
        if m in ("add_us_row", "query_s", "maxrss_mb"):
            a.set_yscale("log")
    ax[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path)


def main():
    d = Path(sys.argv[1])
    rows = build_rows(read_runs(d / "head.jsonl"), read_runs(d / "main.jsonl"))
    write_table(rows, d / "table.txt")
    plot_rows(rows, d / "fig.pdf")
    print((d / "table.txt").read_text())


if __name__ == "__main__":
    main()
