import os
import sys
import time

os.environ.setdefault("RAYON_NUM_THREADS", "1")
import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors

wd = sys.argv[1]
for r, _, fs in os.walk(wd):
    for f in fs:
        fd = os.open(os.path.join(r, f), os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)


def io():
    with open("/proc/self/io") as f:
        return {l.split(":")[0]: int(l.split(":")[1]) for l in f}


def rss():
    with open("/proc/self/status") as f:
        return {
            l.split(":")[0]: int(l.split()[1])
            for l in f
            if l.startswith(("VmRSS", "RssAnon", "RssFile"))
        }


a = io()
t = time.time()
m = EpistemicNearestNeighbors.open(wd)
dt = time.time() - t
b = io()
print(
    f"open_s={dt:.3f} read_MB={(b['read_bytes'] - a['read_bytes']) / 1e6:.1f} write_MB={(b['write_bytes'] - a['write_bytes']) / 1e6:.1f}"
)
Q = np.random.default_rng(7).standard_normal((50, 1000))
ts = []
for i in range(50):
    t = time.time()
    m.neighbors(Q[i : i + 1], 10)
    ts.append(time.time() - t)
c = io()
print(
    f"q_first_s={ts[0]:.3f} q_median_ms={1e3 * np.median(ts):.1f} q_mean_ms={1e3 * np.mean(ts):.1f} qread_MB={(c['read_bytes'] - b['read_bytes']) / 1e6:.0f} rss_kB={rss()}"
)
