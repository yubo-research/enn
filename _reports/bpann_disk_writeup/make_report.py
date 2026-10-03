"""Build data files and the table for bpann_disk_writeup.tex.

- runs/{12d,ranges}_s*.out: ``bench.py TAG SEED 2`` for seeds 0, 2, 4, 6, 8 (10 seeds, run side by side)
- timing_{12d,ranges}.out: ``bench.py TAG 0 1``, run alone

Accuracy values are means ± standard errors over the 10 seeds. Times come from the uncontended seed-0 run:
add time per observation is ``add_s / num_added`` for each checkpoint's segment, fit is ``fit_s`` and query
time per point is ``query_s / 1000``. The seed-0 timing run must reproduce seed 0 of the 10-seed run.

Usage: python reports/bpann_disk_writeup/make_report.py
"""

import glob
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LINE = re.compile(
    r"^seed = (\d+) model = (\w+) n = (\d+) num_added = (\d+) loglik = (\S+) nrmse = (\S+) "
    r"add_s = (\S+) fit_s = (\S+) query_s = (\S+)(?: refits = (\d+) rescales = (\d+) rebuilds = (\d+) learned = (\d))?$",
    re.MULTILINE,
)
MODELS = [("flat", "FLAT"), ("bpann_disk", "BPANN\\_DISK+\\code{NONE}"), ("bpann_disk_auto", "BPANN\\_DISK+\\code{AUTO}")]
TAGS = ("12d", "ranges")
NUM_TEST = 1000
N_GRID = (10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000)


def mean_se(v):
    v = np.asarray(v, dtype=float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else float("nan")


def load(paths):
    """{(model, n): {seed: dict}}"""
    rows = {}
    for p in paths:
        for m in LINE.finditer(open(p).read()):
            seed, model, n, added, ll, nr, add, fit, q, refits, rescales, rebuilds, learned = m.groups()
            rec = dict(added=int(added), loglik=float(ll), nrmse=float(nr), add=float(add), fit=float(fit), q=float(q))
            if refits is not None:
                rec.update(refits=int(refits), rescales=int(rescales), rebuilds=int(rebuilds), learned=int(learned))
            rows.setdefault((model, int(n)), {})[int(seed)] = rec
    return rows


def check_complete(rows, tag, num_seeds):
    for model, _ in MODELS:
        for n in N_GRID:
            got = sorted(rows.get((model, n), {}))
            if len(got) != num_seeds:
                raise ValueError(f"{tag}: {model} n={n} has seeds {got}; is the run complete?")
    return sorted(rows[("flat", N_GRID[0])])


def write_data(tag, data, timing):
    """d_TAG_MODEL.dat: n, loglik, loglik_se, nrmse, nrmse_se (10 seeds); add_us, fit_s, query_ms (seed-0 run)."""
    for model, _ in MODELS:
        out = []
        for n in N_GRID:
            d = data[(model, n)]
            t = timing[(model, n)][0]
            ll, nr = mean_se([r["loglik"] for r in d.values()]), mean_se([r["nrmse"] for r in d.values()])
            out.append([n, *ll, *nr, 1e6 * t["add"] / t["added"], t["fit"], 1e3 * t["q"] / NUM_TEST])
        np.savetxt(
            os.path.join(HERE, f"d_{tag}_{model}.dat"),
            out,
            fmt=["%d"] + ["%.5g"] * 7,
            header="n loglik loglik_se nrmse nrmse_se add_us fit_s query_ms",
            comments="",
        )


def paired(data, a, b, n):
    da, db = data[(a, n)], data[(b, n)]
    diff = [da[s]["loglik"] - db[s]["loglik"] for s in sorted(da)]
    return (*mean_se(diff), int(np.sum(np.array(diff) > 0)), len(diff))


def write_table(data_by_tag, timing_by_tag):
    """At n = 1e6: loglik, nRMSE (10 seeds); add us/obs over the last segment, fit s, query ms/point (seed 0)."""
    lines = []
    n = N_GRID[-1]
    for tag in TAGS:
        name = {"12d": "Irrelevant inputs", "ranges": "Different ranges"}[tag]
        lines.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\emph{{{name}}}}}\\\\")
        for model, label in MODELS:
            d = data_by_tag[tag][(model, n)]
            t = timing_by_tag[tag][(model, n)][0]
            ll, nr = mean_se([r["loglik"] for r in d.values()]), mean_se([r["nrmse"] for r in d.values()])
            lines.append(
                f"{label} & ${ll[0]:+.2f}$ & ${nr[0]:.3f}$ & ${1e6 * t['add'] / t['added']:.1f}$ & ${t['fit']:.2f}$ & "
                f"${1e3 * t['q'] / NUM_TEST:.3f}$\\\\"
            )
    head = ["\\begin{tabular}{@{}lrrrrr@{}}", "\\toprule", "Model & log-lik & nRMSE & add & fit & query \\\\", "\\midrule"]
    with open(os.path.join(HERE, "t_main.tex"), "w") as f:
        f.write("\n".join(head + lines + ["\\bottomrule", "\\end{tabular}"]) + "\n")


def summary(tag, data, timing):
    for n in N_GRID:
        parts = []
        for model, _ in MODELS:
            ll = mean_se([r["loglik"] for r in data[(model, n)].values()])
            nr = mean_se([r["nrmse"] for r in data[(model, n)].values()])
            t = timing[(model, n)][0]
            parts.append(
                f"{model}: ll {ll[0]:+.4f}±{ll[1]:.4f} nr {nr[0]:.4f}±{nr[1]:.4f} "
                f"add {1e6 * t['add'] / t['added']:.2f}us fit {t['fit']:.3f}s q {1e3 * t['q'] / NUM_TEST:.4f}ms"
            )
        pa = paired(data, "bpann_disk_auto", "bpann_disk", n)
        pn = paired(data, "bpann_disk", "flat", n)
        print(f"[{tag}] n={n} | " + " | ".join(parts))
        print(f"[{tag}] n={n}   AUTO-NONE {pa[0]:+.4f}±{pa[1]:.4f} ({pa[2]}/{pa[3]} >0); NONE-FLAT {pn[0]:+.5f}±{pn[1]:.5f}")
        a = timing[("bpann_disk_auto", n)][0]
        if "refits" in a:
            print(f"[{tag}] n={n}   AUTO seed0 refits={a['refits']} rescales={a['rescales']} rebuilds={a['rebuilds']} learned={a['learned']}")
    for model, _ in MODELS:
        tot_add = sum(timing[(model, n)][0]["add"] for n in N_GRID)
        tot_fit = sum(timing[(model, n)][0]["fit"] for n in N_GRID)
        print(f"[{tag}] seed-0 {model}: total add {tot_add:.1f}s total fit {tot_fit:.1f}s")


def main():
    data_by_tag, timing_by_tag = {}, {}
    for tag in TAGS:
        data = load(sorted(glob.glob(os.path.join(HERE, "runs", f"{tag}_s*.out"))))
        seeds = check_complete(data, tag, 10)
        timing = load([os.path.join(HERE, f"timing_{tag}.out")])
        check_complete(timing, tag, 1)
        diff = max(
            abs(timing[k][0][c] - data[k][0][c]) for k in timing for c in ("loglik", "nrmse")
        )
        print(f"[{tag}] seeds={seeds}; timing run vs 10-seed run, seed 0: max |diff| loglik/nrmse = {diff:.2g}")
        write_data(tag, data, timing)
        summary(tag, data, timing)
        data_by_tag[tag], timing_by_tag[tag] = data, timing
    write_table(data_by_tag, timing_by_tag)
    write_macros(timing_by_tag)


def write_macros(timing_by_tag):
    """Anchors (ms per point at n = 1e5) of the ln N reference curves in the query panels."""
    def q(tag, model):
        return 1e3 * timing_by_tag[tag][(model, 100000)][0]["q"] / NUM_TEST
    with open(os.path.join(HERE, "macros.tex"), "w") as f:
        f.write(f"\\newcommand{{\\QREFa}}{{{q('12d', 'bpann_disk'):.5g}}}\n")
        f.write(f"\\newcommand{{\\QREFb}}{{{q('ranges', 'bpann_disk_auto'):.5g}}}\n")


if __name__ == "__main__":
    main()
