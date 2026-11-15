import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D0 = Path(sys.argv[1])
LIDS = (10, 30)
ON, OFF, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
plt.rcParams.update(
    {
        "font.size": 7,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.5,
    }
)


def rows(f):
    p = D0 / f
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def paired(r, a, b):
    x, y = np.array(r[a]), np.array(r[b])
    lr = np.log(x / y)
    se = lr.std(ddof=1) / np.sqrt(len(lr))
    return (
        np.median(x),
        np.median(y),
        np.exp(lr.mean()),
        np.exp(lr.mean() - 1.96 * se),
        np.exp(lr.mean() + 1.96 * se),
    )


def stats():
    out = {}
    for D in LIDS:
        for r in rows(f"lids_D{D}_1G.jsonl"):
            c, w = paired(r, "cold_on", "cold_off"), paired(r, "warm_on", "warm_off")
            pg = (np.median(r["rd_on"]) / 4096, np.median(r["rd_off"]) / 4096)
            out[(D, r["n"])] = (c, w, pg, r["same_mu"])
    return out


def panel_ab(ax, st):
    for j, D in enumerate(LIDS):
        ks = sorted(n for d, n in st if d == D)
        if ks:
            curves(ax[j, 0], st, D, ks)
            ratios(ax[j, 1], st, D, ks)
    for a in ax[:, :2].ravel():
        a.set_xlabel("N (rows)")


def curves(a, st, D, ks):
    n = np.array(ks)
    on = np.array([st[(D, k)][0][0] for k in ks])
    off = np.array([st[(D, k)][0][1] for k in ks])
    a.plot(n, off / 1e3, "o-", color=OFF, ms=3, label="before")
    a.plot(n, on / 1e3, "o-", color=ON, ms=3, label="after")
    a.set_xscale("log")
    a.set_ylabel(f"D={D}: cold query (ms)")
    a.legend(frameon=False, loc="upper left")


def ratios(a, st, D, ks):
    n = np.array(ks)
    for col, idx, lab in ((ON, 0, "cold"), (MUTED, 1, "in RAM")):
        g = np.array([st[(D, k)][idx][2:] for k in ks])
        a.errorbar(
            n,
            g[:, 0],
            yerr=[g[:, 0] - g[:, 1], g[:, 2] - g[:, 0]],
            fmt="o-",
            color=col,
            ms=3,
            capsize=2,
            label=lab,
        )
    a.axhline(1, color=MUTED, lw=0.8, ls=":")
    a.set_xscale("log")
    a.set_ylabel("time ratio after/before")
    a.set_ylim(0.5, 1.6)
    a.legend(frameon=False, loc="upper right")


def panel_keep(a):
    kw = rows("keep_D10_N1000000.jsonl") + rows("keep_D100_N1000000.jsonl")
    labels = ["none", "tree_meta.bin", "tree_codes.bin", "train_x.bin,train_y.bin"]
    for i, D in enumerate((10, 100)):
        v = [
            np.median(
                [
                    u
                    for r in kw
                    if r["D"] == D and r["keep"] == k
                    for u in r["cold_q_us"]
                ]
            )
            / 1e3
            for k in labels
        ]
        a.barh(
            np.arange(4) + 0.4 * i,
            v,
            height=0.38,
            color=("#52514e", "#b9b8b3")[i],
            label=f"D={D}",
        )
    a.set_yticks(np.arange(4) + 0.2, ["nothing", "meta", "codes", "x, y"])
    a.invert_yaxis()
    a.set_xlabel("cold query ms, N=1e6")
    a.set_ylabel("file kept resident")
    a.set_xlim(0, 640)
    a.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.25))


def panel_qd(a):
    q = rows("disk_qd.jsonl")
    for dev, col, lab in (
        ("/home/dsweet/bench_work", "#52514e", "EBS root (store)"),
        ("/mnt/scratch/dsweet_qd", "#1baf7a", "local NVMe"),
    ):
        pts = sorted((r["qd"], r["iops"]) for r in q if r["dir"] == dev)
        a.plot(
            [p[0] for p in pts], [p[1] for p in pts], "o", color=col, ms=3, label=lab
        )
    a.axhline(3000, color=MUTED, lw=0.8, ls=":")
    a.text(1.1, 3300, "3000 IOPS", color=MUTED)
    a.set_xscale("log", base=2)
    a.set_yscale("log")
    a.set_xlabel("reads in flight")
    a.set_ylabel("4 KiB random reads / s")
    a.legend(frameon=False, loc="upper left")


def report(st):
    for k in sorted(st):
        c, w, pg, same = st[k]
        print(
            f"D={k[0]:>3} N={k[1]:>8} cold {c[1] / 1e3:7.1f}->{c[0] / 1e3:7.1f} ms ratio {c[2]:.3f} [{c[3]:.3f},{c[4]:.3f}]"
            f"  warm ratio {w[2]:.3f} [{w[3]:.3f},{w[4]:.3f}]  pages {pg[1]:.0f}->{pg[0]:.0f} same={same}"
        )


def main():
    st = stats()
    fig, ax = plt.subplots(2, 3, figsize=(7.0, 3.9))
    panel_ab(ax, st)
    panel_keep(ax[0, 2])
    panel_qd(ax[1, 2])
    for a, t in zip(ax.ravel(), "abcdef"):
        a.set_title(f"({t})", loc="left", fontsize=7, color=INK)
    fig.tight_layout()
    fig.savefig(D0 / "fig_0236.pdf")
    fig.savefig(D0 / "fig_0236.png", dpi=110)
    report(st)


if __name__ == "__main__":
    main()
