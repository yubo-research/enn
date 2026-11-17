import json
import os
import sys
import time

os.environ.setdefault("RAYON_NUM_THREADS", "1")
import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors


def build_seconds(d, n, work):
    rng = np.random.default_rng(0)
    x0 = rng.random((1, d))
    m = EpistemicNearestNeighbors(x0, x0[:, :1], work_dir=work)
    t0 = time.perf_counter()
    done = 1
    while done < n:
        b = min(1000, n - done)
        x = rng.random((b, d))
        m.add(x, np.sin(3 * x[:, :5]).sum(1, keepdims=True))
        done += b
    m.ensure_index_sync()
    return time.perf_counter() - t0


def main():
    d, n, work, out = (
        int(sys.argv[1]),
        int(float(sys.argv[2])),
        sys.argv[3],
        sys.argv[4],
    )
    dt = build_seconds(d, n, work)
    env = {k: v for k, v in os.environ.items() if k.startswith("ENN_")}
    rec = {
        "arm": os.environ.get("ARM", ""),
        "D": d,
        "N": n,
        "us_per_row": dt / n * 1e6,
        "env": env,
    }
    with open(out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


main()
