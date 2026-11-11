import os
import resource
import time
from pathlib import Path

import numpy as np


def proc_io_counters():
    r = resource.getrusage(resource.RUSAGE_SELF)
    rb = 0
    for line in Path("/proc/self/io").read_text().splitlines():
        if line.startswith("read_bytes"):
            rb = int(line.split()[1])
    return r.ru_minflt, r.ru_majflt, rb


def io_counters():
    d = {}
    for line in Path("/proc/self/io").read_text().splitlines():
        k, _, v = line.partition(":")
        if k in ("read_bytes", "rchar"):
            d[k] = int(v)
    ru = resource.getrusage(resource.RUSAGE_SELF)
    d["majflt"] = ru.ru_majflt
    d["minflt"] = ru.ru_minflt
    return d


def _drop_file(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def drop_cache(wd):
    for root, _, fs in os.walk(wd):
        for f in fs:
            _drop_file(os.path.join(root, f))


def timed_queries(m, qs, params):
    c0 = io_counters()
    ts = []
    for q in qs:
        t = time.perf_counter()
        m.posterior(q[None], params=params)
        ts.append(time.perf_counter() - t)
    c1 = io_counters()
    n = len(qs)
    return (
        float(np.median(ts)) * 1e3,
        float(np.mean(ts)) * 1e3,
        {k: (c1[k] - c0[k]) / n for k in c0},
    )


def gen_sweep(kind, n, r, d, a, w):
    if kind == "gauss":
        x = r.standard_normal((n, d))
        y = x @ w / np.sqrt(d)
    else:
        z = r.standard_normal((n, 4))
        x = z @ a + 0.1 * r.standard_normal((n, d))
        y = np.sin(z[:, 0]) + np.sin(z[:, 1])
    return x, (y + 0.1 * r.standard_normal(n)).reshape(-1, 1)
