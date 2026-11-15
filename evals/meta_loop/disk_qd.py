import json
import mmap
import os
import random
import sys
import threading
import time


def make_file(path, size):
    if os.path.exists(path) and os.path.getsize(path) == size:
        return
    with open(path, "wb") as f:
        blk = os.urandom(2**20)
        f.writelines(blk for _ in range(size // 2**20))
        f.flush()
        os.fsync(f.fileno())


def work(fd, pages, per, seed):
    buf = mmap.mmap(-1, 4096)
    r = random.Random(seed)
    for _ in range(per):
        os.preadv(fd, [buf], r.randrange(pages) * 4096)


def run_qd(path, pages, reads, qd):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    per = reads // qd
    ts = [
        threading.Thread(target=work, args=(fd, pages, per, qd * 100 + i))
        for i in range(qd)
    ]
    t0 = time.perf_counter()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    dt = time.perf_counter() - t0
    os.close(fd)
    n = per * qd
    return {"qd": qd, "us_per_read": 1e6 * dt / n, "iops": n / dt}


def main():
    d = sys.argv[1]
    size = int(sys.argv[2]) * 2**20 if len(sys.argv) > 2 else 2**31
    reads = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
    path = os.path.join(d, "qd_probe.bin")
    make_file(path, size)
    for qd in [int(q) for q in os.environ.get("QDS", "1,4,16,32").split(",")]:
        print(
            json.dumps({"dir": d, **run_qd(path, size // 4096, reads, qd)}), flush=True
        )


if __name__ == "__main__":
    main()
