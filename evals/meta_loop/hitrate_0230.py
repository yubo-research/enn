import json
import os
import sys
import time

import numpy as np
from regions import data_ranges, offsets

rng = np.random.default_rng(11)
COLD = float(os.environ.get("COLD", "665"))


def main():
    fds = [os.open(p, os.O_RDONLY) for p in sys.argv[1:]]
    rs = [data_ranges(fd) for fd in fds]
    alloc = [sum(e - s for s, e in r) for r in rs]
    for fd in fds:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    tot = sum(alloc)
    for _ in range(150000):
        k = rng.choice(len(fds), p=np.array(alloc) / tot)
        os.pread(fds[k], 8192, offsets(rng, rs[k], 1)[0])
    for k, fd in enumerate(fds):
        offs = offsets(rng, rs[k], 4000)
        t = time.perf_counter()
        for o in offs:
            os.pread(fd, 8192, o)
        us = (time.perf_counter() - t) / 4000 * 1e6
        print(
            json.dumps(
                {
                    "path": os.path.basename(sys.argv[1 + k]),
                    "alloc_MB": alloc[k] / 1e6,
                    "us": us,
                    "implied_miss": us / COLD,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
