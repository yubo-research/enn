"""Build data files, table and macros for tree_morph.tex.

- runs/{12d,ranges}_s*.out: ``bench.py TAG SEED 2`` for seeds 0, 2, 4, 6, 8 (10 seeds)
- timing_{12d,ranges}.out: ``bench.py TAG 0 1``, run alone, all four models

Accuracy values are means ± standard errors over the 10 seeds. Times come from the uncontended seed-0 runs:
add time per observation is ``add_s / num_added`` for each checkpoint's segment, fit is ``fit_s`` and query
time per point is ``query_s / 1000``. The seed-0 timing run must reproduce seed 0 of the 10-seed run.

Usage: PYTHONPATH=. python reports/tree_morph/make_report.py
"""

import glob
import os

import numpy as np

from reports.bpann_disk_writeup.make_report import load, mean_se

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = [
    ("flat", "FLAT"),
    ("flat_scale_x", "FLAT+\\code{scale\\_x}"),
    ("bpann_disk", "BPANN\\_DISK+\\code{NONE}"),
    ("bpann_disk_auto", "BPANN\\_DISK+\\code{AUTO}"),
]
TAGS = ("12d", "ranges")
NUM_TEST = 1000
N_GRID = (10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000)


def seeds_of(rows, tag, num_seeds, models):
    """The seeds of a complete run: every (model, n) must have ``num_seeds`` of them."""
    short = [(m, n, sorted(rows.get((m, n), {}))) for m, _ in models for n in N_GRID if len(rows.get((m, n), {})) != num_seeds]
    if short:
        raise ValueError(f"{tag}: incomplete run, e.g. {short[0]}")
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


def not_dominated(vals, higher):
    """True when no other printed number is strictly better."""
    return [not any((o > v if higher else o < v) for j, o in enumerate(vals) if j != i) for i, v in enumerate(vals)]


def paired_not_dominated(seed_maps, higher):
    """True when no other model is strictly better on the same seeds.

    The gap is the paired difference. It counts only when mean ± 1 standard error excludes 0.
    """
    out = []
    for i, mine in enumerate(seed_maps):
        dominated = False
        for j, other in enumerate(seed_maps):
            if i == j:
                continue
            seeds = sorted(set(mine) & set(other))
            diff = [other[s] - mine[s] for s in seeds]
            m, se = mean_se(diff)
            better = (m - se > 0) if higher else (m + se < 0)
            if better:
                dominated = True
                break
        out.append(not dominated)
    return out


def bold_math(expr, best):
    return f"{{\\boldmath{expr}}}" if best else expr


def _table_row(label, stats, flags):
    ll, nr, add, fit, q = stats
    ll_best, nr_best, add_best, fit_best, q_best = flags
    tex_ll = "${:+.3f}\\pm{:.3f}$".format(ll[0], ll[1])
    tex_nr = "${:.4f}\\pm{:.4f}$".format(nr[0], nr[1])
    return (
        f"{label} & {bold_math(tex_ll, ll_best)} "
        f"& {bold_math(tex_nr, nr_best)} "
        "& " + bold_math("${:.1f}$".format(add), add_best) + " "
        "& " + bold_math("${:.2f}$".format(fit), fit_best) + " "
        "& " + bold_math("${:.3f}$".format(q), q_best) + "\\\\"
    )


def write_table(data_by_tag, timing_by_tag):
    """At n = 1e6: loglik, nRMSE (10 seeds); add us/obs over the last segment, fit s, query ms/point (seed 0).

    Within each stream, bold a cell when no other model is strictly better. For log-likelihood and nRMSE the comparison
    is paired across seeds and counts only when mean ± 1 standard error of the difference excludes 0. Times are one run,
    so the best printed number is bold.
    """
    lines = []
    n = N_GRID[-1]

    for tag in TAGS:
        name = {"12d": "Irrelevant inputs", "ranges": "Different ranges"}[tag]
        lines.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\emph{{{name}}}}}\\\\")
        rows = []
        for model, label in MODELS:
            d = data_by_tag[tag][(model, n)]
            ll, nr = mean_se([r["loglik"] for r in d.values()]), mean_se([r["nrmse"] for r in d.values()])
            add, fit, q = per_point(timing_by_tag[tag][(model, n)][0])
            rows.append((label, ll, nr, add, fit, q, {s: r["loglik"] for s, r in d.items()}, {s: r["nrmse"] for s, r in d.items()}))
        ll_best = paired_not_dominated([r[6] for r in rows], True)
        nr_best = paired_not_dominated([r[7] for r in rows], False)
        add_best = not_dominated([r[3] for r in rows], False)
        fit_best = not_dominated([r[4] for r in rows], False)
        q_best = not_dominated([r[5] for r in rows], False)
        print(
            f"[{tag}] bold best: "
            + "; ".join(
                f"{rows[i][0]} ll={ll_best[i]} nr={nr_best[i]} add={add_best[i]} fit={fit_best[i]} q={q_best[i]}"
                for i in range(len(rows))
            )
        )
        for i, (label, ll, nr, add, fit, q, _lls, _nrs) in enumerate(rows):
            lines.append(
                _table_row(
                    label,
                    (ll, nr, add, fit, q),
                    (ll_best[i], nr_best[i], add_best[i], fit_best[i], q_best[i]),
                )
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
            add, fit, q = per_point(timing[(model, n)][0])
            parts.append(f"{model}: ll {ll[0]:+.4f}±{ll[1]:.4f} nr {nr[0]:.4f}±{nr[1]:.4f} add {add:.2f}us fit {fit:.3f}s q {q:.4f}ms")
        print(f"[{tag}] n={n} | " + " | ".join(parts))
        pairs = [
            ("AUTO-NONE", data[("bpann_disk_auto", n)], data[("bpann_disk", n)]),
            ("AUTO-FLATsx", data[("bpann_disk_auto", n)], data[("flat_scale_x", n)]),
            ("FLATsx-FLAT", data[("flat_scale_x", n)], data[("flat", n)]),
            ("NONE-FLAT", data[("bpann_disk", n)], data[("flat", n)]),
        ]
        print(f"[{tag}] n={n}   " + "; ".join(f"{k} {m:+.4f}±{s:.4f} ({c}/{t} >0)" for k, (m, s, c, t) in ((k, paired(a, b)) for k, a, b in pairs)))
        a = timing[("bpann_disk_auto", n)][0]
        print(
            f"[{tag}] n={n}   AUTO seed0 refits={a['refits']} rescales={a['rescales']} rebuilds={a['rebuilds']} "
            f"learned={a['learned']} add {per_point(a)[0]:.2f}us q {per_point(a)[2]:.4f}ms"
        )
    for model, _ in MODELS:
        tot_add = sum(timing[(model, n)][0]["add"] for n in N_GRID)
        tot_fit = sum(timing[(model, n)][0]["fit"] for n in N_GRID)
        print(f"[{tag}] seed-0 {model}: total add {tot_add:.1f}s total fit {tot_fit:.1f}s")


def main():
    data_by_tag, timing_by_tag = {}, {}
    for tag in TAGS:
        data = load(sorted(glob.glob(os.path.join(HERE, "runs", f"{tag}_s*.out"))))
        seeds = seeds_of(data, tag, 10, MODELS)
        timing = load([os.path.join(HERE, f"timing_{tag}.out")])
        seeds_of(timing, tag, 1, MODELS)
        diff = max(abs(timing[k][0][c] - data[k][0][c]) for k in timing for c in ("loglik", "nrmse"))
        print(f"[{tag}] seeds={seeds}; timing run vs 10-seed run, seed 0: max |diff| loglik/nrmse = {diff:.2g}")
        write_data(tag, data, timing, MODELS)
        summary(tag, data, timing)
        data_by_tag[tag], timing_by_tag[tag] = data, timing
    write_table(data_by_tag, timing_by_tag)


if __name__ == "__main__":
    main()
