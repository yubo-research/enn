import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", "1")
import numpy as np
from cold_io import evict, files, pageout, resident_kib, rss_anon_mib, sine_rows, timed

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams, PosteriorFlags

D, MAXN, WORK, OUT, DEV = (
    int(sys.argv[1]),
    int(float(sys.argv[2])),
    sys.argv[3],
    sys.argv[4],
    sys.argv[5],
)
NQ, NA, AG = int(os.environ.get("NQ", "20")), int(os.environ.get("NA", "10")), 32
MODE = os.environ.get("COLD", "pageout")
P = ENNParams(
    k_num_neighbors=10, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.1
)
F = PosteriorFlags(observation_noise=False)


def make(rng, n):
    return sine_rows(rng, n, D)


def go_cold(box):
    box[0].ensure_index_sync()
    if MODE == "pageout":
        pageout(WORK)
        return 0.0
    box[0].persist_index_to_disk()
    box[0] = None
    gc.collect()
    evict(WORK)
    t = time.perf_counter()
    box[0] = EpistemicNearestNeighbors.open(WORK)
    t = 1e6 * (time.perf_counter() - t)
    evict(WORK)
    return t


def cold_queries(box, qx):
    cq, cr, op = [], [], []
    for i in range(NQ):
        op.append(go_cold(box))
        us, rb = timed(DEV, box[0].posterior, qx[i : i + 1], params=P, flags=F)
        cq.append(us)
        cr.append(rb)
    res = resident_kib(WORK)
    wq = [
        timed(DEV, box[0].posterior, qx[i : i + 1], params=P, flags=F)[0]
        for i in range(NQ)
    ]
    return {
        "cold_q_us": cq,
        "cold_q_rd_b": cr,
        "warm_q_us": wq,
        "open_us": op,
        "query_read_set_kib": res,
    }


def cold_adds(box, rng):
    ca, car = [], []
    for _ in range(NA):
        xa, ya = make(rng, AG)
        go_cold(box)
        us, rb = timed(DEV, box[0].add, xa, ya)
        ca.append(us / AG)
        car.append(rb / AG)
    return {"cold_add_us_row": ca, "cold_add_rd_b_row": car}


def record(o):
    skip = ("query_read_set_kib", "store_mib")
    print(
        json.dumps(
            {
                k: (np.median(v) if isinstance(v, list) else v)
                for k, v in o.items()
                if k not in skip
            }
        ),
        flush=True,
    )
    with Path(OUT).open("a") as fh:
        fh.write(json.dumps(o) + "\n")


def main():
    grid = sorted({m * 10**e for e in range(10) for m in (1, 3) if m * 10**e <= MAXN})
    rng = np.random.default_rng(0)
    box = [EpistemicNearestNeighbors(*make(rng, 1), work_dir=WORK)]
    n = 1
    for cp in grid:
        while n < cp:
            b = min(10000, cp - n)
            box[0].add(*make(rng, b))
            n += b
        box[0].ensure_index_sync()
        o = {
            "D": D,
            "n": cp,
            **cold_queries(box, rng.random((NQ, D))),
            **cold_adds(box, rng),
        }
        n += NA * AG
        o["rss_anon_mib"] = rss_anon_mib()
        o["store_mib"] = {f.name: f.stat().st_size / 2**20 for f in files(WORK)}
        record(o)


if __name__ == "__main__":
    main()
