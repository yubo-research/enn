import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

Q = Path(sys.argv[1])
MAIN = Path(
    sys.argv[2] if len(sys.argv) > 2 else "reports/report-0246-data/score/main.jsonl"
)


def load(p):
    return {(o["D"], o["n"], o["seed"]): o for o in map(json.loads, open(p))}


def paired(a, b, m):
    ks = sorted(k for k in a if k in b)
    d = np.array([a[k][m] - b[k][m] for k in ks])
    if len(d) < 2:
        return len(d), float("nan"), float("nan"), float("nan")
    return (
        len(d),
        float(d.mean()),
        float(d.std(ddof=1) / np.sqrt(len(d))),
        float(stats.ttest_1samp(d, 0).pvalue),
    )


def leaves(r):
    w = r.get("walk_totals")
    return w[0] / max(1, w[1]) if w else float("nan")


out = []
for case in sorted(p for p in Q.iterdir() if p.is_dir()):
    arms = {f.stem[1:]: load(f) for f in case.glob("p*.jsonl")}
    base = arms.get("0")
    main = arms.get("main") or {
        k: v for k, v in load(MAIN).items() if base and k in base
    }
    for arm, a in sorted(arms.items(), key=lambda t: (t[0] == "main", t[0])):
        row = {
            "case": case.name,
            "arm": arm,
            "seeds": len(a),
            "loglik": float(np.mean([r["loglik"] for r in a.values()])),
            "nrmse": float(np.mean([r["nrmse"] for r in a.values()])),
            "recall": float(
                np.mean([r.get("recall_at_10", np.nan) for r in a.values()])
            ),
            "query_s": float(np.mean([r["query_s"] for r in a.values()])),
            "leaves_walk": float(np.mean([leaves(r) for r in a.values()])),
        }
        for ref_name, ref in (("p0", base), ("main", main)):
            if ref is None or ref is a:
                continue
            for m in ("loglik", "nrmse"):
                row[f"d{m}_vs_{ref_name}"] = paired(a, ref, m)
            if ref_name == "p0":
                row["query_ratio_vs_p0"] = float(
                    np.mean(
                        [a[k]["query_s"] / ref[k]["query_s"] for k in a if k in ref]
                    )
                )
        out.append(row)
        dl = row.get("dloglik_vs_main", (0, np.nan, np.nan, np.nan))
        dn = row.get("dnrmse_vs_main", (0, np.nan, np.nan, np.nan))
        dl0 = row.get("dloglik_vs_p0", (0, np.nan, np.nan, np.nan))
        print(
            f"{case.name:12s} P={arm:5s} seeds {row['seeds']:2d} leaves/walk {row['leaves_walk']:7.1f} "
            f"recall {row['recall']:.4f} query_s {row['query_s']:.3f} "
            f"x{row.get('query_ratio_vs_p0', 1):.2f} | dloglik vs P0 {dl0[1]:+.4f} (SE {dl0[2]:.4f}, p={dl0[3]:.2g}) "
            f"| vs main dloglik {dl[1]:+.4f} (SE {dl[2]:.4f}, p={dl[3]:.2g}, n={dl[0]}) "
            f"dnrmse {dn[1]:+.5f} (SE {dn[2]:.5f}, p={dn[3]:.2g})"
        )
(Q / "stats.json").write_text(json.dumps(out, indent=1))
