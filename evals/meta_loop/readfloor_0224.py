import json
import os
import sys
import time

import numpy as np


def main():
    f, out, maxgb = sys.argv[1], sys.argv[2], float(sys.argv[3])
    PG = 8192
    fd = os.open(f, os.O_RDWR | os.O_CREAT | os.O_TRUNC)
    rng = np.random.default_rng(0)
    chunk = rng.integers(0, 256, 64 << 20, dtype=np.uint8).tobytes()
    size = 0
    open(out, "w").close()
    sizes = [0.0625 * 2**i for i in range(20)]
    for gb in [s for s in sizes if s <= maxgb]:
        while size < gb * 2**30:
            os.pwrite(fd, chunk, size)
            size += len(chunk)
        os.fsync(fd)
        offs = rng.integers(0, size // PG, 2000) * PG
        t = time.perf_counter()
        for o in offs.tolist():
            os.pread(fd, PG, o)
        us = (time.perf_counter() - t) / len(offs) * 1e6
        rec = {"gb": gb, "pread_us": us}
        print(rec, flush=True)
        with open(out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
