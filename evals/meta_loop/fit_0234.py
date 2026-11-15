import json
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    out, files = sys.argv[1], sys.argv[2:]
    _fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for f in files:
        with open(f) as fh:
            p = [json.loads(l) for l in fh]
        p = [q for q in p if q.get("progress")]
        n = np.array([q["n"] for q in p], float)
        t = np.array([q["add_us_row"] for q in p])
        ax[0].plot(n, t, ".-", lw=0.6, ms=3, label=f.split("_")[-1][:-6])
        ax[1].plot(
            n,
            np.array([q["rd_b_row"] + q["wr_b_row"] for q in p]) / 1e3,
            ".-",
            lw=0.6,
            ms=3,
        )
        for lo, hi in ((1e4, 3e5),):
            m = (n >= lo) & (n <= hi)
            if m.sum() > 2:
                b, a = np.polyfit(np.log(n[m]), t[m], 1)
                print(
                    f,
                    "fit",
                    lo,
                    hi,
                    f"a={a:.1f} b={b:.2f}",
                    "max/plateau ratio %.0f" % (t.max() / t[m].mean()),
                )
    for a_, yl in zip(ax, ("add time (us/row)", "disk KB/row (read+write)")):
        a_.set_xscale("log")
        a_.set_yscale("log")
        a_.set_xlabel("N")
        a_.set_ylabel(yl)
    ax[0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out)


if __name__ == "__main__":
    main()
