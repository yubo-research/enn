import glob
import json
import pathlib
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    D, OUT = sys.argv[1:3]
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.8))
    cols = {1024: "C0", 4096: "C1", 16384: "C2"}
    for f in sorted(glob.glob(f"{D}/long_c*_f16/*.jsonl")):
        c = int(f.split("/")[-2].split("_")[1][1:])
        r = [json.loads(x) for x in pathlib.Path(f).read_text().splitlines()]
        N = np.array([x["N"] for x in r], float)
        for a, k in zip(ax, ("us_per_row", "query_us")):
            t = np.array([x[k] for x in r])
            t = np.convolve(t, np.ones(5) / 5, "valid")
            a.plot(N[2:-2], t, color=cols[c], label=f"{c}-row buffers")
    for a, y in zip(ax, ("add time per row (us)", "query time (us)")):
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlabel("N (rows)")
        a.set_ylabel(y)
        a.grid(alpha=0.3)
        a.axvline(1.1e6, color="gray", ls=":", lw=1)
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT)


if __name__ == "__main__":
    main()
