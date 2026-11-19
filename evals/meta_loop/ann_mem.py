import json
import os
import sys
import time

os.environ.setdefault("RAYON_NUM_THREADS", "1")
import numpy as np
from cold_io import rss_anon_mib

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams, PosteriorFlags


def main():
    work, out, nq = sys.argv[1], sys.argv[2], int(sys.argv[3])
    k = int(os.environ.get("K", "10"))
    rec = {"work": work, "rss_start_mib": rss_anon_mib()}
    m = EpistemicNearestNeighbors.open(work)
    rec["rss_open_mib"] = rss_anon_mib()
    d = int(m._num_dim)
    xt = np.random.default_rng(7).random((nq, d))
    p = ENNParams(
        k_num_neighbors=k, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.01
    )
    f = PosteriorFlags(observation_noise=True)
    t = time.perf_counter()
    for i in range(nq):
        m.posterior(xt[i : i + 1], params=p, flags=f)
    rec["post_ms_q"] = 1e3 * (time.perf_counter() - t) / nq
    rec["rss_after_queries_mib"] = rss_anon_mib()
    rec["index_memory_bytes"] = m.index_memory_bytes()
    rec["n"] = len(m)
    rec["d"] = d
    print(json.dumps(rec), flush=True)
    with open(out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
