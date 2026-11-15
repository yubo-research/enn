import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    with open(sys.argv[1]) as f:
        r = [json.loads(line) for line in f]
    r = [x for x in r if x.get("paused") and x["N"] > 2e6]
    N = [x["N"] for x in r]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot(
        N,
        [x["bufrand_us"] for x in r],
        "o-",
        ms=3,
        c="C0",
        label="buffer file, active adds",
    )
    ax.plot(
        N,
        [x["paused"]["buf_paused_us"] for x in r],
        "s--",
        ms=3,
        c="C0",
        label="buffer file, after fsync+3 s",
    )
    ax.plot(
        N,
        [x["aligned_us"]["leaf_same_offsets"] for x in r],
        "o-",
        ms=3,
        c="C3",
        label="leaf file, active adds",
    )
    ax.plot(
        N,
        [x["paused"]["leaf_paused_us"] for x in r],
        "s--",
        ms=3,
        c="C3",
        label="leaf file, after fsync+3 s",
    )
    ax.set_xscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("random read ($\\mu$s)")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(sys.argv[2])


if __name__ == "__main__":
    main()
