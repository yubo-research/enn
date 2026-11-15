import os

import numpy as np


def data_ranges(fd):
    out, o, size = [], 0, os.fstat(fd).st_size
    while o < size:
        try:
            s = os.lseek(fd, o, os.SEEK_DATA)
        except OSError:
            break
        e = os.lseek(fd, s, os.SEEK_HOLE)
        out.append((s, e))
        o = e
    return out


def offsets(rng, rs, n):
    w = np.array([(e - s) // 8192 for s, e in rs], float)
    i = rng.choice(len(rs), n, p=w / w.sum())
    return [
        rs[k][0] // 8192 * 8192
        + int(rng.integers(0, max(1, (rs[k][1] - rs[k][0]) // 8192))) * 8192
        for k in i
    ]
