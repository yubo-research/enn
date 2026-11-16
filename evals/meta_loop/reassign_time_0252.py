import json
import os
import sys
import time

os.environ.setdefault("RAYON_NUM_THREADS", "1")
import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams, PosteriorFlags


def main():
    D, N, SEED, WORK, OUT = (
        int(sys.argv[1]),
        int(float(sys.argv[2])),
        int(sys.argv[3]),
        sys.argv[4],
        sys.argv[5],
    )
    g = np.random.default_rng(SEED)
    x = g.random((N, D))
    y = np.sin(3 * x[:, :5]).sum(1, keepdims=True) + 0.1 * g.standard_normal((N, 1))
    m = EpistemicNearestNeighbors(x[:1], y[:1], work_dir=WORK)
    t0 = time.perf_counter()
    for i in range(1, N, 1000):
        m.add(x[i : i + 1000], y[i : i + 1000])
    m.ensure_index_sync()
    add_us = (time.perf_counter() - t0) / (N - 1) * 1e6
    q = np.random.default_rng(10_000 + SEED).random((100, D))
    P = ENNParams(
        k_num_neighbors=10, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.1
    )
    F = PosteriorFlags(observation_noise=False)
    m.posterior(q[:5], params=P, flags=F)
    t0 = time.perf_counter()
    m.posterior(q, params=P, flags=F)
    q_ms = (time.perf_counter() - t0) * 1e3 / 100
    rec = {
        "D": D,
        "n": N,
        "seed": SEED,
        "reassign": os.environ.get("ENN_REASSIGN", "0"),
        "every": os.environ.get("ENN_REASSIGN_EVERY", "0"),
        "add_us": add_us,
        "query_ms": q_ms,
    }
    with open(OUT, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(rec)


if __name__ == "__main__":
    main()
