import json
import os
import sys
import time

os.environ.setdefault("RAYON_NUM_THREADS", "1")
import numpy as np
from cold_io import sine_rows

import enn._rust as r
from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams, PosteriorFlags


def per_query(model, qx, p, f):
    r.take_walk_totals()
    r.take_leaf_rows()
    r.take_internal_totals()
    t = time.perf_counter()
    for i in range(len(qx)):
        model.posterior(qx[i : i + 1], params=p, flags=f)
    dt = time.perf_counter() - t
    w = r.take_walk_totals()
    walks = max(w[1] + w[4], 1)
    ip = r.take_internal_totals()
    return {
        "internal_q": ip[0] / len(qx),
        "q_us": 1e6 * dt / len(qx),
        "leaves_q": (w[0] + w[3]) / walks,
        "rows_q": r.take_leaf_rows() / len(qx),
        "walks_q": walks / len(qx),
    }


def build(d, n, work):
    rng = np.random.default_rng(0)
    x, y = sine_rows(rng, 1, d)
    model = EpistemicNearestNeighbors(x, y, work_dir=work)
    r.take_pbuf_stats()
    t = time.perf_counter()
    done = 1
    while done < n:
        b = min(1000, n - done)
        x, y = sine_rows(rng, b, d)
        model.add(x, y)
        done += b
    getattr(model, "ensure_index_sync", lambda: None)()
    return model, 1e6 * (time.perf_counter() - t) / n, r.take_pbuf_stats()


def main():
    d, n, work, out = (
        int(sys.argv[1]),
        int(float(sys.argv[2])),
        sys.argv[3],
        sys.argv[4],
    )
    coef = float(os.environ.get("LEAF_COEF", "0") or 0)
    if coef > 0:
        r.set_leaf_coefficient(coef)
    model, add_us, fl = build(d, n, work)
    p = ENNParams(
        k_num_neighbors=10, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.1
    )
    f = PosteriorFlags(observation_noise=False)
    qx = np.random.default_rng(12345).random((int(os.environ.get("NQ", "300")), d))
    o = {
        "arm": os.environ.get("ARM", ""),
        "D": d,
        "n": n,
        "add_us": add_us,
        "flushes": fl[0],
        "flush_ms": fl[2] / 1e6,
    }
    o.update(per_query(model, qx, p, f))
    print(json.dumps(o), flush=True)
    with open(out, "a") as fh:
        fh.write(json.dumps(o) + "\n")


if __name__ == "__main__":
    main()
