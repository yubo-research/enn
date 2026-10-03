"""Turn complexity_runs/complexity_{12d,ranges}.out into the appendix's data files and table.

Usage: python reports/tree_morph/make_complexity.py   (run from the repo root)

Writes d_cx_{tag}.dat (one row per checkpoint), d_cx_ev_{tag}.dat (batches with an AUTO refit) and
t_complexity.tex, and prints the numbers quoted in the appendix text.
"""

import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TAGS = ("12d", "ranges")
BATCH = 500


def parse(tag):
    batches, checks = [], []
    with open(os.path.join(HERE, "complexity_runs", f"complexity_{tag}.out")) as f:
        for line in f:
            kind, *fields = line.split()
            kv = dict(field.split("=", 1) for field in fields)
            (batches if kind == "B" else checks).append(kv)
    return batches, checks


def segment_add(batches, lo, hi):
    """Mean us/row over all batches in (lo, hi], and median us/row over batches with no AUTO event."""
    seg = [b for b in batches if lo < int(b["n"]) <= hi]
    all_us = [1e6 * float(b["add_s"]) / BATCH for b in seg]
    plain = [u for u, b in zip(all_us, seg) if b["ev"] == "-"]
    return float(np.mean(all_us)), float(np.median(plain))


def checkpoint_row(c, batches, lo):
    n = int(c["n"])
    mean_us, plain_us = segment_add(batches, lo, n)
    per_row = 1024 / n
    return {
        "n": n,
        "add_mean_us": mean_us,
        "add_plain_us": plain_us,
        "query_ms": float(c["query_ms"]),
        "query_same_ms": float(c["query_same_ms"]),
        "ram_B": int(c["anon_kb"]) * per_row,
        "ram_trim_B": int(c["anon_trim_kb"]) * per_row,
        "disk_rows_B": int(c["rows_bytes"]) / n,
        "disk_tree_B": int(c["tree_bytes"]) / n,
        "rescale_ms": float(c["rescale_ms"]),
        "rows_per_leaf": int(c["leaf_rows"]) / int(c["leaves"]),
        "depth": int(c["depth_max"]),
        "budget_frac": min(1.0, math.ceil(128 * math.log(n)) / int(c["leaves"])),
    }


def rows_for(tag):
    batches, checks = parse(tag)
    rows, lo = [], BATCH
    for c in checks:
        rows.append(checkpoint_row(c, batches, lo))
        lo = rows[-1]["n"]
    events = [(int(b["n"]), 1e3 * float(b["add_s"])) for b in batches if "F" in b["ev"]]
    return rows, events


def write_dat(path, header, lines):
    with open(path, "w") as f:
        f.write(" ".join(header) + "\n")
        for line in lines:
            f.write(" ".join(f"{v:.6g}" for v in line) + "\n")


def latex_n(n):
    """1e5 -> $10^{5}$, 3e5 -> $3\\times10^5$."""
    power = int(math.floor(math.log10(n)))
    lead = n // 10**power
    return f"$10^{{{power}}}$" if lead == 1 else f"${lead}\\times10^{power}$"


def write_table(all_rows):
    fmt = (
        "{n} & {add_plain_us:.2f} & {add_mean_us:.1f} & {rescale_ms:.2f} & {query_ms:.3f} & {query_same_ms:.3f}"
        " & {ram_trim_B:.0f} & {disk_rows_B:.0f}+{disk_tree_B:.2f} & {depth} \\\\"
    )
    out = [
        r"\begin{tabular}{@{}rrrrrrrrr@{}}",
        r"\toprule",
        r"$N$ & \multicolumn{2}{c}{add ($\mu$s/row)} & rescale & \multicolumn{2}{c}{query (ms/pt)} & RAM & disk & depth \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){5-6}",
        r" & plain & mean & (ms) & distinct & repeated & (B/row) & (B/row) & \\",
    ]
    for tag, rows in all_rows.items():
        title = "Irrelevant inputs" if tag == "12d" else "Different ranges"
        out += [r"\midrule", rf"\multicolumn{{9}}{{@{{}}l}}{{\emph{{{title}}}}} \\"]
        for r in rows:
            out.append(fmt.format(**{**r, "n": latex_n(r["n"])}))
    out += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(HERE, "t_complexity.tex"), "w") as f:
        f.write("\n".join(out) + "\n")


def report(tag, rows, events):
    print(f"== {tag}")
    for r in rows:
        print(
            "  "
            + " ".join(
                f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in r.items()
            )
        )
    ms = [m for _, m in events]
    print(
        f"  refit batches: {len(events)}; ms min={min(ms):.0f} median={np.median(ms):.0f} max={max(ms):.0f}"
    )
    late = [(n, m) for n, m in events if n > 10**5]
    print("  refit batches above 1e5: " + ", ".join(f"{n}:{m:.0f}ms" for n, m in late))
    same = [(r["n"], r["query_same_ms"]) for r in rows if r["n"] >= 10**5]
    ln = np.log([n for n, _ in same])
    slope, icpt = np.polyfit(ln, [q for _, q in same], 1)
    expo = np.polyfit(ln, np.log([q for _, q in same]), 1)[0]
    print(
        f"  repeated-point query, N>=1e5: affine in ln N slope={slope:.4f} intercept={icpt:.4f}; power {expo:.3f}"
    )
    big = [r for r in rows if r["n"] >= 10**5]
    ns = np.array([r["n"] for r in big], dtype=float)
    ram_b, ram_a = np.polyfit(ns, [r["ram_trim_B"] * r["n"] for r in big], 1)
    resc_b, resc_a = np.polyfit(ns, [r["rescale_ms"] * 1e6 for r in big], 1)
    print(f"  RAM (trimmed), N>=1e5: {ram_b:.1f} B/row + {ram_a / 1e6:.2f} MB")
    print(f"  rescale, N>=1e5: {resc_b:.1f} ns/row + {resc_a / 1e6:.2f} ms")


def main():
    all_rows = {}
    for tag in TAGS:
        rows, events = rows_for(tag)
        all_rows[tag] = rows
        keys = list(rows[0])
        write_dat(
            os.path.join(HERE, f"d_cx_{tag}.dat"),
            keys,
            [[r[k] for k in keys] for r in rows],
        )
        write_dat(os.path.join(HERE, f"d_cx_ev_{tag}.dat"), ["n", "ms"], events)
        report(tag, rows, events)
    write_table(all_rows)


if __name__ == "__main__":
    main()
