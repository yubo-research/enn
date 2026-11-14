import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    S, Q, OUT = sys.argv[1:4]
    C = {"main": "#888888", "head": "#d95f02", "gate": "#1b9e77"}
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for l in ("main", "head", "gate"):
        try:
            r = [
                json.loads(x)
                for x in Path(f"{S}/{l}_D100.jsonl").read_text().splitlines()
            ]
        except FileNotFoundError:
            continue
        r = [x for x in r if "add_us_row" in x and x["n"] >= 1000]
        ax[0].plot(
            [x["n"] for x in r],
            [x["add_us_row"] for x in r],
            "o-",
            ms=3,
            color=C[l],
            label=l,
        )
    ax[0].set(
        xscale="log",
        yscale="log",
        xlabel="N",
        ylabel="add, µs/row",
        title="(a) add time, D=100, 1 GB cap",
    )
    ax[0].legend(fontsize=7)
    h = pd.read_json(f"{Q}/head.jsonl", lines=True)
    m = pd.read_json(f"{Q}/main.jsonl", lines=True)
    g = pd.read_json(f"{Q}/gate.jsonl", lines=True)
    for i, (col, t) in enumerate((("loglik", "(b) loglik"), ("nrmse", "(c) NRMSE"))):
        for l, d in (("main", m), ("head", h), ("gate", g)):
            for D, ls in ((10, "-"), (100, "--")):
                s = d[d.D == D].groupby("n")[col].mean()
                ax[1 + i].plot(
                    s.index,
                    s.values,
                    ls,
                    marker="o",
                    ms=3,
                    color=C[l],
                    label=f"{l} D={D}" if i == 0 else None,
                )
        ax[1 + i].set(xscale="log", xlabel="N", title=t)
    for a in ax:
        a.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(OUT)


if __name__ == "__main__":
    main()
