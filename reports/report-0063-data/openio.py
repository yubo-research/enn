import os
import resource
import sys
import time

os.environ.setdefault("RAYON_NUM_THREADS", "1")
from enn.enn.enn_class import EpistemicNearestNeighbors

wd = sys.argv[1]
for r, _, fs in os.walk(wd):
    for f in fs:
        fd = os.open(os.path.join(r, f), os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)


def io():
    return {l.split(":")[0]: int(l.split(":")[1]) for l in open("/proc/self/io")}


a = io()
t = time.time()
m = EpistemicNearestNeighbors.open(wd)
dt = time.time() - t
b = io()
ru = resource.getrusage(resource.RUSAGE_SELF)
print(
    f"open_s={dt:.1f} read_MB={(b['read_bytes'] - a['read_bytes']) / 1e6:.0f} write_MB={(b['write_bytes'] - a['write_bytes']) / 1e6:.0f} user={ru.ru_utime:.1f} sys={ru.ru_stime:.1f} majflt={ru.ru_majflt}"
)
