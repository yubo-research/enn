import glob
import json
import sys
from collections import defaultdict

import numpy as np


def load(d, cap, arm):
    out = defaultdict(list)
    for f in sorted(glob.glob(f"{d}/rep*/stall_D100_{cap}_pb{arm}.jsonl")):
        with open(f) as fh:
            for r in map(json.loads, fh):
                out[r["n"]].append(r)
    return out


def dirty(d, arm):
    tx, idx, cgd = [], [], []
    for f in sorted(glob.glob(f"{d}/rep*/dirty_{arm}.jsonl")):
        with open(f) as fh:
            recs = [json.loads(l) for l in fh]
        for r in recs:
            fs = r["files"]
            tx.append(sum(v[1] for k, v in fs.items() if k.startswith("train_")))
            idx.append(sum(v[1] for k, v in fs.items() if k.startswith("index/")))
            if "cg_dirty" in r:
                cgd.append(r["cg_dirty"])
    m = lambda a: round(float(np.mean(a)), 1) if a else None
    return m(tx), m(idx), m(cgd)


def main():
    d, cap, arms = sys.argv[1], sys.argv[2], sys.argv[3:]
    data = {a: load(d, cap, a) for a in arms}
    keys = ("wall_us_row", "cpu_us_row", "wr_b_row", "rd_b_row", "dirty_mib")
    for a in arms:
        print(f"== {a}  dirty MiB (train_*, index/*, cgroup): {dirty(d, a)}")
        for n in sorted(data[a]):
            rs = data[a][n]
            vals = [np.mean([r[k] for r in rs]) for k in keys]
            sp = np.ptp([r["wall_us_row"] for r in rs]) if len(rs) > 1 else 0
            print(
                f"  n={n:>9}  "
                + "  ".join(f"{k.split('_')[0]}={v:8.1f}" for k, v in zip(keys, vals))
                + f"  wall_range={sp:.1f} reps={len(rs)}"
            )
    base = data[arms[0]]
    for a in arms[1:]:
        r = [
            np.mean([x["wall_us_row"] for x in data[a][n]])
            / np.mean([x["wall_us_row"] for x in base[n]])
            for n in sorted(base)
            if n in data[a]
        ]
        print(f"ratio {a}/{arms[0]} wall per checkpoint:", [round(x, 2) for x in r])


if __name__ == "__main__":
    main()
