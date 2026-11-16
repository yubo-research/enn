import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

Q = Path(sys.argv[1])
R = Path(__file__).resolve().parents[2] / "reports"
MAINS = {
    "d10": R / "report-0246-data/score/main.jsonl",
    "d100i5": R / "report-0251-data/qual/d100i5_1e5/pmain.jsonl",
}


def load(p):
    return {(o["D"], o["n"], o["seed"]): o for o in map(json.loads, open(p))}


def paired(a, b, m):
    ks = sorted(k for k in a if k in b and m in a[k] and m in b[k])
    d = np.array([a[k][m] - b[k][m] for k in ks])
    if len(d) < 2:
        return [len(d), float("nan"), float("nan"), float("nan")]
    return [
        len(d),
        float(d.mean()),
        float(d.std(ddof=1) / np.sqrt(len(d))),
        float(stats.ttest_1samp(d, 0).pvalue),
    ]


def ratio(a, b, m):
    ks = [k for k in a if k in b]
    return float(np.mean([a[k][m] / b[k][m] for k in ks])) if ks else float("nan")


out = []
for case in sorted(p for p in Q.iterdir() if p.is_dir()):
    arms = {f.stem[1:]: load(f) for f in case.glob("r*.jsonl")}
    base = arms.get("0")
    mp = next(
        (
            MAINS[k]
            for k in sorted(MAINS, key=len, reverse=True)
            if case.name.startswith(k)
        ),
        None,
    )
    main = load(mp) if mp and mp.exists() else {}
    for arm, a in sorted(arms.items(), key=lambda t: int(t[0])):
        row = {"case": case.name, "arm": arm, "seeds": len(a)}
        for m in ("loglik", "nrmse", "recall_at_10", "query_s", "add_us_row"):
            row[m] = float(np.mean([r.get(m, np.nan) for r in a.values()]))
        for ref_name, ref in (("r0", base), ("main", main)):
            if not ref or ref is a:
                continue
            for m in ("loglik", "nrmse", "recall_at_10"):
                row[f"d{m}_vs_{ref_name}"] = paired(a, ref, m)
            row[f"query_ratio_vs_{ref_name}"] = ratio(a, ref, "query_s")
            row[f"add_ratio_vs_{ref_name}"] = ratio(a, ref, "add_us_row")
        out.append(row)
        s = f"{case.name:12s} M={arm:3s} n={row['seeds']:2d} recall {row['recall_at_10']:.4f} query_s {row['query_s']:.3f} add_us {row['add_us_row']:.2f}"
        for ref_name in ("r0", "main"):
            if f"dloglik_vs_{ref_name}" in row:
                dl, dn = row[f"dloglik_vs_{ref_name}"], row[f"dnrmse_vs_{ref_name}"]
                s += (
                    f" | vs {ref_name}: dloglik {dl[1]:+.4f} (SE {dl[2]:.4f}, p={dl[3]:.2g}) dnrmse {dn[1]:+.5f} "
                    f"(p={dn[3]:.2g}) query x{row[f'query_ratio_vs_{ref_name}']:.2f} add x{row[f'add_ratio_vs_{ref_name}']:.2f}"
                )
        print(s)
(Q / "stats.json").write_text(json.dumps(out, indent=1))
