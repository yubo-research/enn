import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

D, OUT = sys.argv[1], sys.argv[2]
COL = {"c64": "#4C72B0", "c256": "#DD8452"}
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.1))
lines = []
DIM = "?"
for f in sorted(glob.glob(os.path.join(D, "capq_*.jsonl"))):
    tag = f.rsplit("_pb0_", 1)[1].removesuffix(".jsonl")
    arm = tag.split("_")[0]
    with open(f) as fh:
        rows = [json.loads(x) for x in fh]
    rows = [r for r in rows if r["n"] >= 10]
    DIM = rows[0]["D"] if rows else DIM
    n = [r["n"] for r in rows]
    style = {
        "color": COL.get(arm, "gray"),
        "marker": "o",
        "ms": 3,
        "lw": 1.2,
        "ls": "-" if "big" in tag or tag.endswith("r1") else "--",
        "label": tag,
    }
    ax[0].plot(n, [r["add_us_row"] for r in rows], **style)
    ax[1].plot(n, [r["q_us_mean"] / 1e3 for r in rows], **style)
    ax[2].plot(n, [max(r["q_rd_ops"], 0.1) for r in rows], **style)
    for r in rows:
        if r["n"] >= 1e5:
            lines.append(
                f"{tag} n={r['n']:.0e} add={r['add_us_row']:.1f}us q={r['q_us_mean'] / 1e3:.2f}ms "
                f"q_rd_ops={r['q_rd_ops']:.1f} codes_res={r['res_mib'].get('tree_codes.bin', 0):.0f}MiB"
            )
for a, t in zip(
    ax,
    [
        "add (us/row; lower is better)",
        "posterior (ms/query; lower is better)",
        "device reads / query (lower is better)",
    ],
):
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("N")
    a.set_title(t, fontsize=9)
    a.grid(alpha=0.3, which="both", lw=0.4)
ax[0].legend(fontsize=6, frameon=False)
fig.suptitle(
    f"256 MB cap, D={DIM}, 1 CPU, EBS: leaf capacity 64 (blue) vs 256 (orange)",
    fontsize=9,
)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}.{ext}", dpi=160)
with open(f"{OUT}.txt", "w") as fh:
    fh.write("\n".join(lines) + "\n")
print("\n".join(lines))

if len(sys.argv) > 4:
    import numpy as np

    Q, C64_1E6 = sys.argv[3], sys.argv[4]

    def load(p):
        return {o["seed"]: o for o in map(json.loads, open(p))}

    cases = [
        ("D=10\n1e5", f"{Q}/d10_1e5/c256.jsonl", f"{Q}/d10_1e5/c64.jsonl"),
        (
            "D=100 (5-dim)\n1e5",
            f"{Q}/d100i5_1e5/c256.jsonl",
            f"{Q}/d100i5_1e5/c64.jsonl",
        ),
        ("D=10\n1e6", f"{Q}/d10_1e6/c256.jsonl", C64_1E6),
    ]
    fig2, bx = plt.subplots(2, 1, figsize=(3.4, 4.6))
    w = 0.38
    for i, (lab, fa, fb) in enumerate(cases):
        a, b = load(fa), load(fb)
        ks = [k for k in a if k in b]
        for j, (m, c) in enumerate((("add_us_row", "#55A868"), ("query_s", "#C44E52"))):
            r = np.mean([a[k][m] for k in ks]) / np.mean([b[k][m] for k in ks])
            bx[0].bar(
                i + (j - 0.5) * w,
                r,
                w,
                color=c,
                label=("add", "query")[j] if i == 0 else None,
            )
            bx[0].text(i + (j - 0.5) * w, r + 0.03, f"{r:.2f}", ha="center", fontsize=7)
        d = np.array([a[k]["loglik"] - b[k]["loglik"] for k in ks])
        bx[1].errorbar(
            i,
            d.mean(),
            yerr=d.std(ddof=1) / np.sqrt(len(ks)),
            fmt="o",
            color="k",
            capsize=3,
        )
    bx[0].axhline(1, color="gray", lw=0.8)
    bx[0].set_xticks(range(3), [c[0] for c in cases], fontsize=7)
    bx[0].set_title(
        "in-RAM time ratio, 256-row / 64-row leaves\n(below 1 = 256-row faster)",
        fontsize=8,
    )
    bx[0].legend(fontsize=7, frameon=False)
    bx[1].axhline(0, color="gray", lw=0.8)
    bx[1].set_xlim(-0.5, 2.5)
    bx[1].set_xticks(range(3), [c[0] for c in cases], fontsize=7)
    bx[1].set_title(
        "loglik(256) $-$ loglik(64), mean $\\pm$ SE\n(higher is better)", fontsize=8
    )
    fig2.tight_layout()
    for ext in ("pdf", "png"):
        fig2.savefig(f"{OUT}_ram.{ext}", dpi=160)
