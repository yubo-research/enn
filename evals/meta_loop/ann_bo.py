import json
import os
import sys
import time

os.environ.setdefault("RAYON_NUM_THREADS", "1")
import numpy as np
from cold_io import rss_anon_mib

import enn._rust as R
from enn import create_optimizer, turbo_enn_config
from enn.turbo.config import ENNSurrogateConfig

D, ITERS, WORK, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
ARMS = int(os.environ.get("ARMS", "1"))
SEED = int(os.environ.get("SEED", "0"))
WEIGHTING = bool(int(os.environ.get("WEIGHTING", "0")))
PRIOR_N = int(float(os.environ.get("PRIOR_N", "0")))


def objective(x):
    return np.sin(3 * x[:, : min(5, x.shape[1])]).sum(1, keepdims=True)


def drain():
    R.take_walk_totals()
    R.take_exact_rows()
    R.take_internal_cache_stats()


def create_seeded_optimizer():
    config = turbo_enn_config(enn=ENNSurrogateConfig(k=10, weighting=WEIGHTING))
    opt = create_optimizer(
        bounds=np.array([[0.0, 1.0]] * D), config=config, seed=SEED, work_dir=WORK
    )
    if PRIOR_N:
        g = np.random.default_rng(1000 + SEED)
        xp = g.random((PRIOR_N, D))
        opt.tell(xp, objective(xp) + 0.1 * g.standard_normal((PRIOR_N, 1)))
    return opt


def main():
    opt = create_seeded_optimizer()
    n = PRIOR_N
    t0 = time.perf_counter()
    for it in range(ITERS):
        drain()
        t = time.perf_counter()
        x = opt.ask(num_arms=ARMS)
        t_ask = time.perf_counter() - t
        w_ask = R.take_walk_totals()
        e_ask = R.take_exact_rows()
        c_ask = R.take_internal_cache_stats()
        y = objective(x)
        t = time.perf_counter()
        opt.tell(x, y)
        t_tell = time.perf_counter() - t
        w_tell = R.take_walk_totals()
        e_tell = R.take_exact_rows()
        n += len(x)
        tm = opt.telemetry()
        rec = {
            "D": D,
            "it": it,
            "n": n,
            "arms": ARMS,
            "weighting": WEIGHTING,
            "ask_s": t_ask,
            "tell_s": t_tell,
            "dt_fit": tm.dt_fit,
            "dt_gen": tm.dt_gen,
            "dt_sel": tm.dt_sel,
            "dt_tell": tm.dt_tell,
            "num_candidates": tm.num_candidates,
            "ask_walks": w_ask[1] + w_ask[4],
            "ask_leaves": w_ask[0] + w_ask[3],
            "ask_exact_rows": e_ask,
            "ask_icache_hit_miss": list(c_ask),
            "ask_budget_stops": w_ask[2] + w_ask[5],
            "tell_walks": w_tell[1] + w_tell[4],
            "tell_leaves": w_tell[0] + w_tell[3],
            "tell_exact_rows": e_tell,
            "rss_anon_mib": rss_anon_mib(),
            "wall_s": time.perf_counter() - t0,
        }
        if it % 50 == 0 or it == ITERS - 1:
            print(json.dumps(rec), flush=True)
        with open(OUT, "a") as fh:
            fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
