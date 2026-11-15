import json
import os
import shutil
import sys
import time

import numpy as np
from regions import data_ranges, offsets

D = sys.argv[1]
rng = np.random.default_rng(7)


def timeit(path, n=3000):
    fd = os.open(path, os.O_RDONLY)
    rs = data_ranges(fd)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    offs = offsets(rng, rs, n)
    t = time.perf_counter()
    for o in offs:
        os.pread(fd, 8192, o)
    us = (time.perf_counter() - t) / n * 1e6
    os.close(fd)
    return {
        "path": os.path.basename(path),
        "alloc_MB": sum(e - s for s, e in rs) / 1e6,
        "n_ranges": len(rs),
        "us": us,
    }


def main():
    shutil.copyfile(f"{D}/buf.bin", f"{D}/buf_copy.bin")
    os.sync()
    for _ in range(3):
        for p in ("buf.bin", "buf_copy.bin", "leaf.bin"):
            print(json.dumps(timeit(f"{D}/{p}")), flush=True)


if __name__ == "__main__":
    main()
