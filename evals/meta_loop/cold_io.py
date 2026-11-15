import ctypes
import os
import subprocess
import time
from pathlib import Path

import numpy as np

_libc = ctypes.CDLL(None, use_errno=True)
_libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
MADV_PAGEOUT = 21


def disk(dev):
    rows = (l.split() for l in Path("/proc/diskstats").read_text().splitlines())
    hit = next((f for f in rows if f[2] == dev), None)
    if hit is None:
        raise SystemExit("device not found")
    return int(hit[5]) * 512, int(hit[9]) * 512


def files(work):
    return [f for f in Path(work).rglob("*") if f.is_file()]


def evict(work):
    for f in files(work):
        fd = os.open(f, os.O_RDONLY)
        try:
            os.fsync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def pageout(work):
    root = os.path.realpath(work)
    for l in Path("/proc/self/maps").read_text().splitlines():
        f = l.split(maxsplit=5)
        if len(f) == 6 and f[5].startswith(root):
            a, b = (int(v, 16) for v in f[0].split("-"))
            _libc.madvise(a, b - a, MADV_PAGEOUT)
    evict(work)


def _fincore_kib(path):
    r = subprocess.run(
        ["fincore", "-b", "-n", "-o", "RES", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return int(r.stdout.split()[0]) / 1024
    except (ValueError, IndexError):
        return 0.0


def resident_kib(work):
    out = {}
    for f in files(work):
        v = _fincore_kib(f)
        if v:
            out[f.name] = out.get(f.name, 0) + v
    return out


def rss_anon_mib():
    for l in Path("/proc/self/status").read_text().splitlines():
        if l.startswith("RssAnon"):
            return int(l.split()[1]) / 1024
    return 0.0


def timed(dev, fn, *a, **kw):
    r0, _ = disk(dev)
    t = time.perf_counter()
    fn(*a, **kw)
    return 1e6 * (time.perf_counter() - t), disk(dev)[0] - r0


def sine_rows(rng, n, d):
    x = rng.random((n, d))
    noise = 0.1 * rng.standard_normal((n, 1))
    return x, noise + np.sin(3 * x[:, : min(5, d)]).sum(1, keepdims=True)
