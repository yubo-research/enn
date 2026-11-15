import json
import os
import sys
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", "1")
import numpy as np
from cold_io import MADV_PAGEOUT, _libc, files, sine_rows, timed

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams, PosteriorFlags

D, N, WORK, OUT, DEV = (
    int(sys.argv[1]),
    int(float(sys.argv[2])),
    sys.argv[3],
    sys.argv[4],
    sys.argv[5],
)
NQ = int(os.environ.get("NQ", "20"))
KEEPS = os.environ.get(
    "KEEPS", "none;tree_meta.bin;tree_codes.bin;train_x.bin,train_y.bin"
).split(";")
P = ENNParams(
    k_num_neighbors=10, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.1
)
F = PosteriorFlags(observation_noise=False)


def _settle(p, keep):
    fd = os.open(p, os.O_RDONLY)
    try:
        _touch_or_drop(fd, p.name in keep)
    finally:
        os.close(fd)


def _touch_or_drop(fd, keep):
    if keep:
        while os.read(fd, 1 << 20):
            pass
        return
    os.fsync(fd)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)


def pageout_except(keep):
    root = os.path.realpath(WORK)
    for l in Path("/proc/self/maps").read_text().splitlines():
        f = l.split(maxsplit=5)
        if len(f) == 6 and f[5].startswith(root) and Path(f[5]).name not in keep:
            a, b = (int(v, 16) for v in f[0].split("-"))
            _libc.madvise(a, b - a, MADV_PAGEOUT)
    for p in files(WORK):
        _settle(p, keep)


def main():
    rng = np.random.default_rng(0)
    m = EpistemicNearestNeighbors(*sine_rows(rng, 1, D), work_dir=WORK)
    n = 1
    while n < N:
        b = min(10000, N - n)
        m.add(*sine_rows(rng, b, D))
        n += b
    m.ensure_index_sync()
    qx = rng.random((NQ, D))
    for rep in range(2):
        for ks in KEEPS:
            keep = set() if ks == "none" else set(ks.split(","))
            us, rb = [], []
            for i in range(NQ):
                pageout_except(keep)
                t, r = timed(DEV, m.posterior, qx[i : i + 1], params=P, flags=F)
                us.append(t)
                rb.append(r)
            o = {
                "D": D,
                "n": N,
                "rep": rep,
                "keep": ks,
                "cold_q_us": us,
                "cold_q_rd_b": rb,
            }
            print(
                json.dumps(
                    {
                        "keep": ks,
                        "rep": rep,
                        "ms": float(np.median(us)) / 1e3,
                        "pages": float(np.median(rb)) / 4096,
                    }
                ),
                flush=True,
            )
            with Path(OUT).open("a") as fh:
                fh.write(json.dumps(o) + "\n")


if __name__ == "__main__":
    main()
