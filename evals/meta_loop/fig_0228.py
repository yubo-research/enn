import glob
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "reports/report-0228-data"


def last(fs, n=4):
    with open(glob.glob(f"{D}/hyb_d2_fs{fs}/*.jsonl")[0]) as f:
        rs = [json.loads(line) for line in f]
    rs = [r for r in rs if r["N"] > 7e6][-n:]
    g = lambda f: sum(f(r) for r in rs) / len(rs)
    return {
        "q": g(lambda r: r["query_us"]),
        "leaf": g(lambda r: r["leaf_only_us"]),
        "l1": g(
            lambda r: (
                r["per_level_calls_us_bytes"]["1"][1]
                * r["per_level_calls_us_bytes"]["1"][0]
            )
        ),
        "l2": g(
            lambda r: (
                r["per_level_calls_us_bytes"]["2"][1]
                * r["per_level_calls_us_bytes"]["2"][0]
            )
        ),
        "al": g(lambda r: r["aligned_us"]["al8k"]),
        "same": g(lambda r: r["aligned_us"]["leaf_same_offsets"]),
        "rnd": g(lambda r: r["bufrand_us"]),
    }


a, b = last(64), last(16)
fig, ax = plt.subplots(1, 2, figsize=(7, 2.8))
for i, (nm, d) in enumerate((("fence 64 rows", a), ("fence 16 rows", b))):
    rest = d["q"] - d["l1"] - d["l2"] - d["leaf"]
    bot = 0
    for v, c, lb in (
        (d["leaf"], "#4c72b0", "leaf read (leaf-only probe)"),
        (d["l1"], "#dd8452", "level-1 buffer"),
        (d["l2"], "#55a868", "level-2 buffer"),
        (rest, "#bbbbbb", "other (RAM levels, CPU)"),
    ):
        ax[0].bar(i, v, bottom=bot, color=c, label=lb if i == 0 else None)
        bot += v
ax[0].set_xticks([0, 1])
ax[0].set_xticklabels(["fence 64", "fence 16"])
ax[0].set_ylabel("query time (us)")
ax[0].legend(fontsize=6)
ax[0].set_title("(a) query time by part, N~7.5e6", fontsize=8)
names = [
    "leaf.bin\nrandom",
    "leaf.bin,\nbuffer offsets",
    "buf.bin\n8 KB aligned",
    "buf.bin\n7 KB",
]
vals = [a["leaf"], a["same"], a["al"], a["rnd"]]
ax[1].bar(names, vals, color=["#4c72b0", "#4c72b0", "#dd8452", "#dd8452"])
ax[1].tick_params(labelsize=6)
ax[1].set_ylabel("one random pread (us)")
ax[1].set_title("(b) same cap, different file", fontsize=8)
plt.tight_layout()
plt.savefig(f"{D}/fig.pdf")
print(a, b)
