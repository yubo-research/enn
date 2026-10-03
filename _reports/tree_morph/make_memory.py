"""Turn memory_runs/memory_12d.out (memory_probe.py 16000000) into t_memory.tex and print the numbers quoted in store.tex.

Usage: python reports/tree_morph/make_memory.py   (run from the repo root)
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST_ROW = 1_000_000


def parse():
    rows = []
    with open(os.path.join(HERE, "memory_runs", "memory_12d.out")) as f:
        for line in f:
            kv = dict(field.split("=", 1) for field in line.split() if "=" in field and not field.startswith("top"))
            rows.append({k: float(v) for k, v in kv.items()})
    return rows


def millions(n):
    return f"{n / 1e6:g}"


def main():
    rows = parse()
    lines = [
        r"\begin{tabular}{@{}rrrrrrr@{}}",
        r"\toprule",
        r"$N$ & anon. & file & peak & add & refits & query \\",
        r"(M) & (MiB) & (MiB) & (MiB) & (s/M) & & (ms) \\",
        r"\midrule",
    ]
    prev = None
    for r in rows:
        if prev is not None and r["n"] >= FIRST_ROW:
            dn = (r["n"] - prev["n"]) / 1e6
            add = (r["add_s_cum"] - prev["add_s_cum"]) / dn
            refits = int(r["refits"] - prev["refits"])
            lines.append(
                f"{millions(r['n'])} & {r['anon_mb']:.0f} & {r['file_mb']:.0f} & {r['hwm_mb']:.0f} & {add:.1f} & "
                f"{refits} & {r['query_ms']:.3f} \\\\"
            )
            print(
                f"n={r['n']:.0f} anon={r['anon_mb']:.1f} file={r['file_mb']:.1f} peak={r['hwm_mb']:.1f} "
                f"add_s_per_M={add:.2f} refits+={refits} rescales={r['rescales']:.0f} "
                f"repartitions={r['repartitions']:.0f} query_ms={r['query_ms']:.3f}"
            )
        prev = r
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(HERE, "t_memory.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    below = [r for r in rows if r["n"] <= 10_000_000 and r["anon_mb"] > 50]
    lo, hi = below[0], below[-1]
    slope = (hi["anon_mb"] - lo["anon_mb"]) * 1024 * 1024 / (hi["n"] - lo["n"])
    print(f"anonymous growth {lo['n']:.0f}..{hi['n']:.0f}: {slope:.1f} B/row")


if __name__ == "__main__":
    main()
