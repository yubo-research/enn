"""Summarize stream_mbpann.py results at each checkpoint (mean and std over seeds).

Usage: python reports/mbpann/analyze_mbpann.py RESULTS.json OUT_PREFIX
Writes OUT_PREFIX.md and OUT_PREFIX.tex.
"""

import json
import sys
from collections import defaultdict

import numpy as np

COLS = [
    ("loo", "LOO", "{:.3f}"),
    ("test", "test", "{:.3f}"),
    ("recall", "recall", "{:.2f}"),
    ("metric_sec", "metric s", "{:.2f}"),
    ("add_sec", "add s", "{:.2f}"),
    ("query_sec", "query s", "{:.2f}"),
]


def group(rows, n):
    out = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["n"] == n:
            for key, _, _ in COLS + [("loo_exact", "", ""), ("test_exact", "", "")]:
                out[(r["problem"], r["method"])][key].append(r[key])
    return out


def cell(vals, fmt):
    v = np.asarray(vals, dtype=float)
    return f"{fmt.format(v.mean())} ± {fmt.format(v.std())}" if len(v) > 1 else fmt.format(v.mean())


def table_rows(rows, n):
    g = group(rows, n)
    lines = []
    for (prob, meth), vals in g.items():
        ref = f"{np.mean(vals['loo_exact']):.3f} / {np.mean(vals['test_exact']):.3f}"
        lines.append([prob, meth] + [cell(vals[k], f) for k, _, f in COLS] + [ref])
    return lines


def main(path, prefix):
    rows = json.load(open(path))
    n = max(r["n"] for r in rows)
    head = ["problem", "method"] + [h for _, h, _ in COLS] + ["exact-nbr LOO / test"]
    body = table_rows(rows, n)
    md = [f"n = {n}, seeds = {len({r['seed'] for r in rows})}", "", "| " + " | ".join(head) + " |"]
    md.append("|" + "---|" * len(head))
    md += ["| " + " | ".join(r) + " |" for r in body]
    open(prefix + ".md", "w").write("\n".join(md) + "\n")
    tex = [" & ".join(r).replace("±", r"$\pm$").replace("_", r"\_") + r" \\" for r in body]
    open(prefix + ".tex", "w").write("\n".join(tex) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
