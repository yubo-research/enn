import json
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    with open(sys.argv[1]) as fh:
        r = [json.loads(ln) for ln in fh]
    N = np.array([x["N"] for x in r])
    q = np.array([x["query_us"] for x in r])
    l = np.array([x["leaf_only_us"] for x in r])
    np.array([x["preads_per_query"] for x in r])
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax[0].plot(
        N, q, ".", ms=3, color="#c0392b", label="buffered query (mrg, 4096 rows)"
    )
    ax[0].plot(N, l, ".", ms=3, color="#2c7fb8", label="leaf-only: one 8 KB pread")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].axvline(1.5e6, color="gray", ls=":", lw=1)
    ax[0].set_xlabel("N (rows)")
    ax[0].set_ylabel("query time (µs)")
    ax[0].legend(fontsize=7)
    ax[0].set_title("(a) 256M cap, D-free model", fontsize=9)
    ax[1].plot(N, q / l, ".", ms=3, color="#555555")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("N (rows)")
    ax[1].set_ylabel("buffered / leaf-only")
    ax[1].set_title("(b) overhead of buffer reads", fontsize=9)
    fig.tight_layout()
    fig.savefig(sys.argv[2])


if __name__ == "__main__":
    main()
