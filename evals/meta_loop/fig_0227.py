import glob
import json
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    D = sys.argv[1]
    runs = {
        "all disk (0225)": "reports/report-0225-data/long_c4096_f16/bs_buffered_256M_c4096_f16.jsonl"
    }
    for d in sorted(glob.glob(f"{D}/hyb_d*")):
        runs[
            f"disk levels<={d[-1]}" + (" (killed at 1.1e7)" if d[-1] == "1" else "")
        ] = glob.glob(d + "/*.jsonl")[0]
    rngs = [(0, 1e6), (1e6, 1e7), (1e7, 3.1e7)]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6))
    print("run | metric | <1e6 | 1e6-1e7 | 1e7-3e7 | slope(us/lnN, N>=1e7)")
    for (name, f), c in zip(runs.items(), ["#555", "#c0392b", "#2c7fb8", "#27ae60"]):
        with open(f) as fh:
            r = [json.loads(l) for l in fh]
        N = np.array([x["N"] for x in r])
        for a, key in zip(ax, ["query_us", "us_per_row", "preads_per_query"]):
            v = np.array([x[key] for x in r])
            a.plot(N, v, ".", ms=3, color=c, label=name)
            m = [v[(N > lo) & (N <= hi)].mean() for lo, hi in rngs]
            s = N >= 1e7
            sl = np.polyfit(np.log(N[s]), v[s], 1)[0]
            print(
                f"{name} | {key} | "
                + " | ".join(f"{x:.1f}" for x in m)
                + f" | {sl:.1f}"
            )
    for a, t in zip(
        ax,
        ["query (us, lower better)", "add (us/row, lower better)", "preads per query"],
    ):
        a.set_xscale("log")
        a.set_yscale("log" if "pread" not in t else "linear")
        a.set_xlabel("N")
        a.set_title(t, fontsize=8)
    ax[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(sys.argv[2])


if __name__ == "__main__":
    main()
