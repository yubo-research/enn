import json
import os
import resource
import sys
import time

os.environ.setdefault("RAYON_NUM_THREADS", "1")
import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors


def main():
    D, N, W = 100, 100000, sys.argv[1]
    rng = np.random.default_rng(1)

    def mk(n):
        return rng.random((n, D)), rng.random((n, 1))

    m = EpistemicNearestNeighbors(*mk(1), work_dir=W)
    r0 = resource.getrusage(resource.RUSAGE_SELF)
    t0 = time.perf_counter()
    for _ in range(N // 1000):
        m.add(*mk(1000))
    m.ensure_index_sync()
    r1 = resource.getrusage(resource.RUSAGE_SELF)
    print(
        json.dumps(
            {
                "arm": os.environ["ARM"],
                "us_row": 1e6 * (time.perf_counter() - t0) / N,
                "minflt_row": (r1.ru_minflt - r0.ru_minflt) / N,
                "majflt": r1.ru_majflt - r0.ru_majflt,
                "utime": r1.ru_utime - r0.ru_utime,
                "stime": r1.ru_stime - r0.ru_stime,
            }
        )
    )


if __name__ == "__main__":
    main()
