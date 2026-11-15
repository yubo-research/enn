import json
import mmap
import os
import random
import sys
import time

from disk_qd import make_file


def run(path, size, s_kib, reads, seed):
    s = s_kib * 1024
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    buf = mmap.mmap(-1, s)
    r = random.Random(seed)
    slots = size // s
    lat = []
    for _ in range(reads):
        off = r.randrange(slots) * s
        t0 = time.perf_counter()
        os.preadv(fd, [buf], off)
        lat.append(time.perf_counter() - t0)
    os.close(fd)
    lat.sort()
    mean = sum(lat) / len(lat)
    return {
        "kib": s_kib,
        "us_per_read": 1e6 * mean,
        "p50_us": 1e6 * lat[len(lat) // 2],
        "p90_us": 1e6 * lat[int(0.9 * len(lat))],
        "mb_s": s / mean / 1e6,
        "us_per_4k_page": 1e6 * mean / (s_kib / 4),
    }


def main():
    d = sys.argv[1]
    size = (int(sys.argv[2]) if len(sys.argv) > 2 else 4096) * 2**20
    reads = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    path = os.path.join(d, "extent_probe.bin")
    make_file(path, size)
    sizes = [
        int(x)
        for x in os.environ.get("SIZES_KIB", "4,8,16,32,64,128,256,512,1024").split(",")
    ]
    for rep in range(int(os.environ.get("REPS", "2"))):
        for i, s in enumerate(sizes):
            print(
                json.dumps(
                    {"dir": d, "rep": rep, **run(path, size, s, reads, 1000 * rep + i)}
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
