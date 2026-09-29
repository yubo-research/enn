"""Build data files, table and macros for tree_morph.tex.

- runs/{12d,ranges}_s*.out: ``bench.py TAG SEED 2`` for seeds 0, 2, 4, 6, 8 (10 seeds, run side by side)
- timing_{12d,ranges}.out: ``bench.py TAG 0 1``, run alone
- ../bpann_disk_writeup/{runs/*,timing_*.out}: the same protocol, seeds and data, run on the code before the morph
  (BPANN_DISK+AUTO re-partitioned by a background bulk build and swap). Used as the "previous AUTO" reference.

Accuracy values are means ± standard errors over the 10 seeds. Times come from the uncontended seed-0 runs:
add time per observation is ``add_s / num_added`` for each checkpoint's segment, fit is ``fit_s`` and query
time per point is ``query_s / 1000``. The seed-0 timing run must reproduce seed 0 of the 10-seed run.

Usage: python reports/tree_morph/make_report.py
"""

import glob
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OLD = os.path.join(HERE, "..", "bpann_disk_writeup")
LINE = re.compile(
    r"^seed = (\d+) model = (\w+) n = (\d+) num_added = (\d+) loglik = (\S+) nrmse = (\S+) "
    r"add_s = (\S+) fit_s = (\S+) query_s = (\S+)(?: refits = (\d+) rescales = (\d+) rebuilds = (\d+) learned = (\d))?$",
    re.MULTILINE,
)
MODELS = [
    ("flat", "FLAT"),
    ("flat_scale_x", "FLAT+\\code{scale\\_x}"),
    ("bpann_disk", "BPANN\\_DISK+\\code{NONE}"),
    ("bpann_disk_auto", "BPANN\\_DISK+\\code{AUTO}"),
]
OLD_MODELS = [("flat", ""), ("bpann_disk", ""), ("bpann_disk_auto", "")]
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


def check_complete(rows, tag, num_seeds, models):
    for model, _ in models:
        for n in N_GRID:
            got = sorted(rows.get((model, n), {}))
            if len(got) != num_seeds:
                raise ValueError(f"{tag}: {model} n={n} has seeds {got}; is the run complete?")
    return sorted(rows[("flat", N_GRID[0])])


def per_point(t):
    return 1e6 * t["add"] / t["added"], t["fit"], 1e3 * t["q"] / NUM_TEST


def write_data(tag, data, timing, models, suffix=""):
    """d_TAG_MODEL.dat: n, loglik, loglik_se, nrmse, nrmse_se (10 seeds); add_us, fit_s, query_ms (seed-0 run)."""
    for model, _ in models:
        out = []
        for n in N_GRID:
            d = data[(model, n)]
            ll, nr = mean_se([r["loglik"] for r in d.values()]), mean_se([r["nrmse"] for r in d.values()])
            out.append([n, *ll, *nr, *per_point(timing[(model, n)][0])])
        np.savetxt(
            os.path.join(HERE, f"d_{tag}_{model}{suffix}.dat"),
            out,
            fmt=["%d"] + ["%.5g"] * 7,
            header="n loglik loglik_se nrmse nrmse_se add_us fit_s query_ms",
            comments="",
        )


def paired(da, db, key="loglik"):
    """Mean ± se of the same-seed difference a - b, and how many seeds have a > b."""
    diff = [da[s][key] - db[s][key] for s in sorted(da)]
    return (*mean_se(diff), int(np.sum(np.array(diff) > 0)), len(diff))


def write_table(data_by_tag, timing_by_tag, old_by_tag, old_timing_by_tag):
    """At n = 1e6: loglik, nRMSE (10 seeds); add us/obs over the last segment, fit s, query ms/point (seed 0)."""
    lines = []
    n = N_GRID[-1]

    def row(label, d, t):
        ll, nr = mean_se([r["loglik"] for r in d.values()]), mean_se([r["nrmse"] for r in d.values()])
        add, fit, q = per_point(t)
        return f"{label} & ${ll[0]:+.3f}$ & ${nr[0]:.4f}$ & ${add:.1f}$ & ${fit:.2f}$ & ${q:.3f}$\\\\"

    for tag in TAGS:
        name = {"12d": "Irrelevant inputs", "ranges": "Different ranges"}[tag]
        lines.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\emph{{{name}}}}}\\\\")
        for model, label in MODELS:
            lines.append(row(label, data_by_tag[tag][(model, n)], timing_by_tag[tag][(model, n)][0]))
        key = ("bpann_disk_auto", n)
        lines.append(
            row("\\quad previous \\code{AUTO}$^\\dagger$", old_by_tag[tag][key], old_timing_by_tag[tag][key][0])
        )
    head = ["\\begin{tabular}{@{}lrrrrr@{}}", "\\toprule", "Model & log-lik & nRMSE & add & fit & query \\\\", "\\midrule"]
    with open(os.path.join(HERE, "t_main.tex"), "w") as f:
        f.write("\n".join(head + lines + ["\\bottomrule", "\\end{tabular}"]) + "\n")


def summary(tag, data, timing, old, old_timing):
    for n in N_GRID:
        parts = []
        for model, _ in MODELS:
            ll = mean_se([r["loglik"] for r in data[(model, n)].values()])
            nr = mean_se([r["nrmse"] for r in data[(model, n)].values()])
            add, fit, q = per_point(timing[(model, n)][0])
            parts.append(f"{model}: ll {ll[0]:+.4f}±{ll[1]:.4f} nr {nr[0]:.4f}±{nr[1]:.4f} add {add:.2f}us fit {fit:.3f}s q {q:.4f}ms")
        print(f"[{tag}] n={n} | " + " | ".join(parts))
        pairs = [
            ("AUTO-NONE", data[("bpann_disk_auto", n)], data[("bpann_disk", n)]),
            ("AUTO-FLATsx", data[("bpann_disk_auto", n)], data[("flat_scale_x", n)]),
            ("FLATsx-FLAT", data[("flat_scale_x", n)], data[("flat", n)]),
            ("NONE-FLAT", data[("bpann_disk", n)], data[("flat", n)]),
            ("AUTOnew-AUTOold", data[("bpann_disk_auto", n)], old[("bpann_disk_auto", n)]),
            ("NONEnew-NONEold", data[("bpann_disk", n)], old[("bpann_disk", n)]),
        ]
        print(f"[{tag}] n={n}   " + "; ".join(f"{k} {m:+.4f}±{s:.4f} ({c}/{t} >0)" for k, (m, s, c, t) in ((k, paired(a, b)) for k, a, b in pairs)))
        nrd = paired(data[("bpann_disk_auto", n)], old[("bpann_disk_auto", n)], "nrmse")
        print(f"[{tag}] n={n}   nRMSE AUTOnew-AUTOold {nrd[0]:+.5f}±{nrd[1]:.5f} ({nrd[2]}/{nrd[3]} >0)")
        a, o = timing[("bpann_disk_auto", n)][0], old_timing[("bpann_disk_auto", n)][0]
        print(
            f"[{tag}] n={n}   AUTO seed0 new refits={a['refits']} rescales={a['rescales']} rebuilds={a['rebuilds']} "
            f"learned={a['learned']} add {per_point(a)[0]:.2f}us q {per_point(a)[2]:.4f}ms | old add {per_point(o)[0]:.2f}us "
            f"q {per_point(o)[2]:.4f}ms rebuilds={o.get('rebuilds')}"
        )
    for model, _ in MODELS:
        tot_add = sum(timing[(model, n)][0]["add"] for n in N_GRID)
        tot_fit = sum(timing[(model, n)][0]["fit"] for n in N_GRID)
        print(f"[{tag}] seed-0 {model}: total add {tot_add:.1f}s total fit {tot_fit:.1f}s")
    tot_old = sum(old_timing[("bpann_disk_auto", n)][0]["add"] for n in N_GRID)
    print(f"[{tag}] seed-0 previous bpann_disk_auto: total add {tot_old:.1f}s")


def main():
    data_by_tag, timing_by_tag, old_by_tag, old_timing_by_tag = {}, {}, {}, {}
    for tag in TAGS:
        data = load(sorted(glob.glob(os.path.join(HERE, "runs", f"{tag}_s*.out"))))
        seeds = check_complete(data, tag, 10, MODELS)
        timing = load([os.path.join(HERE, f"timing_{tag}.out")])
        check_complete(timing, tag, 1, MODELS)
        diff = max(abs(timing[k][0][c] - data[k][0][c]) for k in timing for c in ("loglik", "nrmse"))
        print(f"[{tag}] seeds={seeds}; timing run vs 10-seed run, seed 0: max |diff| loglik/nrmse = {diff:.2g}")
        old = load(sorted(glob.glob(os.path.join(OLD, "runs", f"{tag}_s*.out"))))
        old_seeds = check_complete(old, tag, 10, OLD_MODELS)
        old_timing = load([os.path.join(OLD, f"timing_{tag}.out")])
        check_complete(old_timing, tag, 1, OLD_MODELS)
        if old_seeds != seeds:
            raise ValueError(f"{tag}: previous-code seeds {old_seeds} differ from {seeds}")
        write_data(tag, data, timing, MODELS)
        write_data(tag, old, old_timing, [("bpann_disk_auto", "")], suffix="_old")
        summary(tag, data, timing, old, old_timing)
        data_by_tag[tag], timing_by_tag[tag] = data, timing
        old_by_tag[tag], old_timing_by_tag[tag] = old, old_timing
    write_table(data_by_tag, timing_by_tag, old_by_tag, old_timing_by_tag)


if __name__ == "__main__":
    main()
