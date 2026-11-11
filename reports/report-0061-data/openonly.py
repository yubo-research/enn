import os
import sys

os.environ.setdefault("RAYON_NUM_THREADS", "1")
from enn.enn.enn_class import EpistemicNearestNeighbors

wd = sys.argv[1]
for r, _, fs in os.walk(wd):
    for f in fs:
        fd = os.open(os.path.join(r, f), os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
m = EpistemicNearestNeighbors.open(wd)
