import json
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    D = sys.argv[1]
    with open(f"{D}/frag.jsonl") as fh:
        cold = [json.loads(l) for l in fh if l.startswith("{")]
    cm = {}
    for r in cold:
        cm.setdefault(r["path"], []).append(r["us"])
    warm, cur, mixed = {}, None, {}
    with open(f"{D}/hit.txt") as fh:
        hit_lines = fh.readlines()
    for l in hit_lines:
        if l.startswith("=="):
            cur = len(l.split()) - 1
        elif l.startswith("{") and cur:
            r = json.loads(l)
            (warm if cur == 1 else mixed)[r["path"]] = r["us"]
    _fig, ax = plt.subplots(1, 2, figsize=(7, 2.8), sharey=True)
    n = ["leaf.bin", "buf.bin", "buf_copy.bin"]
    ax[0].bar(
        range(3),
        [np.mean(cm[k][1:]) for k in n],
        color=["#4c78a8", "#f58518", "#54a24b"],
    )
    ax[0].set_xticks(range(3), ["leaf", "buf", "buf\n(seq. copy)"])
    ax[0].set_title("(a) cold reads (cache dropped)")
    ax[0].set_ylabel("random 8 KB read (us)")
    x = np.arange(2)
    ax[1].bar(
        x - 0.2,
        [warm["leaf.bin"], warm["buf.bin"]],
        0.4,
        label="file alone",
        color="#4c78a8",
    )
    ax[1].bar(
        x + 0.2,
        [mixed["leaf.bin"], mixed["buf.bin"]],
        0.4,
        label="both files share cache",
        color="#e45756",
    )
    ax[1].set_xticks(x, ["leaf", "buf"])
    ax[1].set_title("(b) warmed 256M cache")
    ax[1].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{D}/fig.pdf")


if __name__ == "__main__":
    main()
