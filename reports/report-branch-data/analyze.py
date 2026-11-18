import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

H = Path(__file__).resolve().parent
ARMS = {"ov_brhead": ("branch", "#d55e00"), "ov_0260main": ("main", "#0072b2")}
out = {}


def load(p):
    return [json.loads(line) for line in open(p)] if Path(p).exists() else []


def fit(n, y):
    n, y = np.asarray(n, float), np.asarray(y, float)
    m = (n >= 1e3) & (n <= 1e6)
    if m.sum() < 3:
        return float("nan"), float("nan")
    b, a = np.polyfit(np.log(n[m]), y[m], 1)
    return a, b


def pexp(n, y):
    n, y = np.asarray(n, float), np.asarray(y, float)
    m = (n >= 1e4) & (n <= 1e6)
    return (
        float(np.polyfit(np.log(n[m]), np.log(y[m]), 1)[0])
        if m.sum() >= 3
        else float("nan")
    )


cfgs = [(10, "256M"), (100, "256M"), (10, "1G"), (100, "1G")]
fig, ax = plt.subplots(len(cfgs), 4, figsize=(15, 3.1 * len(cfgs)), squeeze=False)
tab = []
for r, (D, cap) in enumerate(cfgs):
    for ov, (name, col) in ARMS.items():
        rows = load(H / f"cap/capq_D{D}_{cap}_{ov}.jsonl")
        rows2 = load(H / f"cap2/capq_D{D}_{cap}_{ov}_x.jsonl")
        if rows2 and rows2[-1]["n"] > (rows[-1]["n"] if rows else 0):
            rows = rows2
        if not rows:
            continue
        n = [o["n"] for o in rows]
        add = [o["add_us_row"] for o in rows]
        q = [o["q_us_med"] / 1e3 for o in rows]
        rss = [o["rss_anon_mib"] for o in rows]
        wall = [o["wall_s"] for o in rows]
        disk = None
        ax[r][0].loglog(n[1:], add[1:], "o-", c=col, label=name, ms=3)
        ax[r][1].loglog(n, q, "o-", c=col, ms=3)
        ax[r][2].semilogx(n, rss, "o-", c=col, ms=3)
        ax[r][3].loglog(n, wall, "o-", c=col, ms=3)
        ns = np.array(n)

        def at(v, x, ns=ns):
            return (
                float(np.interp(np.log(x), np.log(ns), v))
                if ns.min() <= x <= ns.max()
                else float("nan")
            )

        for lab, v in (("add", add), ("query", q)):
            a, b = fit(n, v)
            tab.append(
                {
                    "D": D,
                    "cap": cap,
                    "arm": name,
                    "metric": lab,
                    "a": a,
                    "b": b,
                    "pexp_1e4_1e6": pexp(n, v),
                    "v1e5": at(v, 1e5),
                    "v1e6": at(v, 1e6),
                    "v3e6": at(v, 3e6),
                    "step_1e6_3e6": at(v, 3e6) / at(v, 1e6),
                    "G_3e6_1e5": at(v, 3e6) / at(v, 1e5),
                }
            )
        tab.append(
            {
                "D": D,
                "cap": cap,
                "arm": name,
                "metric": "rss_anon_mib",
                "max": max(rss),
                "min": min(rss),
            }
        )
        tab.append(
            {
                "D": D,
                "cap": cap,
                "arm": name,
                "metric": "wall_s",
                "maxn": int(ns.max()),
                "wall": wall[-1],
            }
        )
    ax[r][0].set_ylabel(f"D={D}, cap {cap}\nadd µs/row")
    ax[r][1].set_ylabel("query median ms")
    ax[r][2].set_ylabel("RssAnon MiB")
    ax[r][3].set_ylabel("cumulative wall s")
    for c in range(4):
        ax[r][c].set_xlabel("N rows")
        ax[r][c].grid(alpha=0.3)
ax[0][0].legend()
fig.tight_layout()
fig.savefig(H / "fig_scaling.pdf")
out["scaling"] = tab

sizes = {}
for line in (
    (H / "cap/sweep.log").read_text().splitlines()
    if (H / "cap/sweep.log").exists()
    else []
):
    if line.startswith("STORE"):
        sizes[line] = 1
out["store_lines"] = list(sizes)

qual = []
cases = (
    sorted(p for p in (H / "qual").iterdir() if p.is_dir())
    if (H / "qual").exists()
    else []
)
for c in cases:
    a = {o["seed"]: o for o in load(c / "r1.jsonl")}
    b = {o["seed"]: o for o in load(c / "r0.jsonl")}
    ks = sorted(set(a) & set(b))
    row = {"case": c.name, "seeds": len(ks)}
    for m in (
        "nrmse",
        "loglik",
        "recall_at_10",
        "knn_nrmse_exact",
        "query_s",
        "add_us_row",
        "maxrss_mb",
    ):
        x = np.array([a[k][m] for k in ks])
        y = np.array([b[k][m] for k in ks])
        d = x - y
        row[m] = {"branch": float(x.mean()), "main": float(y.mean())}
        if len(d) > 1:
            row[m].update(
                diff=float(d.mean()),
                se=float(d.std(ddof=1) / np.sqrt(len(d))),
                p=float(stats.ttest_1samp(d, 0).pvalue),
                ratio=float(np.mean(x / y)),
            )
    qual.append(row)
out["quality"] = qual
(H / "results.json").write_text(json.dumps(out, indent=1))

if qual:
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.3))
    for j, (m, lab) in enumerate(
        (
            ("nrmse", "ΔNRMSE (branch − main)"),
            ("loglik", "Δloglik (branch − main)"),
            ("recall_at_10", "recall@10: branch − main"),
        )
    ):
        for dd, mk in ((10, "o"), (100, "s")):
            xs, ys, es = [], [], []
            for r in qual:
                if f"D{dd}_" in r["case"] and "se" in r[m]:
                    xs.append(int(r["case"].split("_n")[1]))
                    ys.append(r[m]["diff"])
                    es.append(2 * r[m]["se"])
            ax[j].errorbar(
                xs,
                ys,
                es,
                fmt=mk + "-",
                capsize=3,
                label=f"D={dd}" + (", INTRINSIC=5" if dd == 100 else ""),
            )
        ax[j].axhline(0, c="k", lw=0.7)
        ax[j].set_xscale("log")
        ax[j].set_xlabel("N rows")
        ax[j].set_ylabel(lab)
        ax[j].grid(alpha=0.3)
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(H / "fig_quality.pdf")
print(json.dumps(out, indent=1)[:6000])
