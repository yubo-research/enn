"""Build data files and the table for the morph micro-benchmark section of tree_morph.tex.

- morph_runs/{new,old,none}_{N0}_{SHRINK}.out: ``morph_bench N0 60000 uniform SHRINK [nomorph]``, run one at a time
  with nothing else running. ``new`` and ``none`` use the current code, ``old`` the re-insertion morph it replaced.
- morph_runs/none_{N0}_1.0.out: the same with no metric change (control).

Each line reports, after ``end - N0`` rows added since the change: q300 (seconds for 300 queries, k = 10),
recall@10 on 100 of them against brute force, add time per row over the last segment and whether the morph is running.
The ``fresh`` line is a tree bulk-built over all rows at the end. In the data files the state just after the change
is plotted at ``START_X`` thousand adds, so that it fits on a log axis.

Usage: python reports/tree_morph/make_bench.py
"""

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "morph_runs")
LINE = re.compile(
    r"^\s*(start|morph|fresh) end=\s*(\d+) pages=\s*(\d+) q300=(\S+)s recall=(\S+) add_us/row=(\S+) "
    r"worst_batch=(\S+)s morphing=(\w+)$",
    re.MULTILINE,
)
ARMS = ("none", "old", "new")
CASES = [(100000, "0.001"), (400000, "0.001"), (100000, "0.25"), (400000, "0.25")]
START_X = 0.5
MORPH_LABEL = {"old": "re-insertion", "new": "reference moves"}
SCALE_LABEL = {"0.001": "$\\times 10^{-3}$", "0.25": "$\\times\\tfrac14$"}


def load(arm, n0, shrink):
    rows, fresh = [], None
    for kind, end, pages, q, rec, add, worst, morphing in LINE.findall(open(os.path.join(RUNS, f"{arm}_{n0}_{shrink}.out")).read()):
        r = dict(k=(int(end) - n0) / 1000, pages=int(pages), q=float(q), recall=float(rec), add=float(add),
                 worst=float(worst), morphing=morphing == "true")
        if kind == "fresh":
            fresh = r
        else:
            rows.append(r)
    return rows, fresh


def window(rows):
    """(last checkpoint with the morph running, first without), in thousands of adds."""
    running = [r["k"] for r in rows if r["morphing"] and r["k"] > 0]
    after = [r["k"] for r in rows if not r["morphing"] and r["k"] > 0]
    return (max(running) if running else 0), (min(after) if after else float("nan"))


def at(rows, k):
    return next(r for r in rows if r["k"] == k)


def write_curve(path, rows):
    with open(path, "w") as f:
        f.write("k q recall add_us\n")
        for r in rows:
            f.write(f"{START_X if r['k'] == 0 else r['k']:g} {r['q']:.5g} {r['recall']:.4f} {r['add']:.4g}\n")


def morph_row(arm, n0, shrink, rows, fresh):
    """Print the numbers quoted in the text and return this arm's table row."""
    in_window = [r["add"] for r in rows if r["k"] > 0 and r["morphing"]]
    lo, hi = window(rows)
    end = rows[-1]
    print(
        f"{arm} N0={n0} shrink={shrink}: recall +1k {at(rows, 1)['recall']:.3f} +5k {at(rows, 5)['recall']:.3f}; "
        f"q300 +1k {at(rows, 1)['q']:.4f} +5k {at(rows, 5)['q']:.4f} end {end['q']:.4f} (fresh {fresh['q']:.4f}, "
        f"x{end['q'] / fresh['q']:.2f}); window ends in ({lo:g}k, {hi:g}k]; in-window add {min(in_window):.0f}-"
        f"{max(in_window):.0f} us/row; after {end['add']:.2f}"
    )
    return (
        f"{n0 // 1000}k & {SCALE_LABEL[shrink]} & {MORPH_LABEL[arm]} & ${at(rows, 1)['recall']:.3f}$ & "
        f"${at(rows, 5)['q'] * 1e3:.1f}$ & ${end['q'] / fresh['q']:.2f}$ & ${lo:g}$--${hi:g}$ & ${max(in_window):.0f}$\\\\"
    )


def case_rows(n0, shrink):
    """Write the curves of one case and the fresh-tree reference line; return its table rows."""
    tag = f"{n0 // 1000}k_{shrink.replace('.', 'p')}"
    table, freshes = [], []
    for arm in ARMS:
        rows, fresh = load(arm, n0, shrink)
        freshes.append(fresh["q"])
        write_curve(os.path.join(HERE, f"d_mb_{arm}_{tag}.dat"), rows)
        if arm != "none":
            table.append(morph_row(arm, n0, shrink, rows, fresh))
        else:
            print(f"none N0={n0} shrink={shrink}: start q300 {rows[0]['q']:.4f} recall {rows[0]['recall']:.3f}; "
                  f"end q300 {rows[-1]['q']:.4f} recall {rows[-1]['recall']:.3f}")
            last_k = rows[-1]["k"]
    q = sum(freshes) / len(freshes)
    with open(os.path.join(HERE, f"d_mb_fresh_{tag}.dat"), "w") as f:
        f.write(f"k q\n{START_X:g} {q:.5g}\n{last_k:g} {q:.5g}\n")
    return table


def print_control():
    for n0 in (100000, 400000):
        rows, fresh = load("none", n0, "1.0")
        print(f"control (no change) N0={n0}: start q300 {rows[0]['q']:.4f}, end {rows[-1]['q']:.4f}, fresh {fresh['q']:.4f} "
              f"(x{rows[-1]['q'] / fresh['q']:.2f})")


def main():
    blocks = ["\n".join(case_rows(n0, shrink)) for n0, shrink in CASES]
    print_control()
    with open(os.path.join(HERE, "t_bench.tex"), "w") as f:
        f.write("\n\\addlinespace[2pt]\n".join(blocks) + "\n")


if __name__ == "__main__":
    main()
