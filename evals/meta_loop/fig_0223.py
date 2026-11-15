import glob
import json
import pathlib
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(f):
    r = [json.loads(x) for x in pathlib.Path(f).read_text().splitlines()]
    return np.array([x["N"] for x in r], float), r


def main():
    d0, d1, out = sys.argv[1:4]
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9))
    for c, col in ((4096, "C1"), (16384, "C2")):
        for d, ls, nm in ((d0, "--", "scan"), (d1, "-", "indexed")):
            fs = glob.glob(f"{d}/long_c{c}_f16/*.jsonl")
            if not fs:
                continue
            N, r = load(fs[0])
            for a, k in zip(ax, ("us_per_row", "query_us")):
                t = np.convolve([x[k] for x in r], np.ones(5) / 5, "valid")
                a.plot(N[2:-2], t, color=col, ls=ls, label=f"{c} rows, {nm}")
    for a, y in zip(ax, ("add time per row (us)", "query time (us)")):
        a.set(xscale="log", yscale="log", xlabel="N (rows)", ylabel=y)
        a.grid(alpha=0.3)
        a.axvline(1.1e6, color="gray", ls=":", lw=1)
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out)


if __name__ == "__main__":
    main()
