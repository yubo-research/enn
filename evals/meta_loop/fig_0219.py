import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():

    rows = [json.loads(l) for l in Path(sys.argv[1]).read_text().splitlines()]
    p = [r for r in rows if r.get("progress") and "res_mib" in r]
    n = [r["n"] / 1e6 for r in p]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ck = [r for r in rows if not r.get("progress")]
    ax[0].plot(
        [r["n"] / 1e6 for r in ck],
        [r["add_us_row"] for r in ck],
        "o-",
        color="#0072B2",
        label="checkpoint",
    )
    ax[0].plot(n, [r["add_us_row"] for r in p], ".", color="#D55E00", label="interval")
    ax[0].set_yscale("log")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("N (millions)")
    ax[0].set_ylabel("add time (us/row)")
    ax[0].legend(frameon=False)
    cols = {
        "tree_codes.bin": "#0072B2",
        "tree_slots.bin": "#009E73",
        "tree_meta.bin": "#CC79A7",
        "train_x.bin": "#E69F00",
        "train_y.bin": "#999999",
    }
    for f, c in cols.items():
        ax[1].plot(
            n,
            [r["res_mib"].get(f, 0) for r in p],
            "-",
            color=c,
            label=f.replace(".bin", ""),
        )
    ax[1].axhline(256, ls=":", color="k")
    ax[1].text(n[0], 240, "cap 256 MiB", va="top", fontsize=7)
    ax[1].set_xlabel("N (millions)")
    ax[1].set_ylabel("resident in page cache (MiB)")
    ax[1].legend(frameon=False, fontsize=7)
    for a, t in zip(ax, "ab"):
        a.set_title(t, loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(sys.argv[2])


if __name__ == "__main__":
    main()
