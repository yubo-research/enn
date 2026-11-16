import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resident_map import resident

from enn.enn.enn_class import EpistemicNearestNeighbors


def main():
    w, n_target, out = sys.argv[1], int(float(sys.argv[2])), sys.argv[3]
    d, rng = 100, np.random.default_rng(0)
    m = EpistemicNearestNeighbors(rng.random((1, d)), rng.random((1, 1)), work_dir=w)
    n = 1
    while n < n_target:
        m.add(rng.random((1000, d)), rng.random((1000, 1)))
        n += 1000
    r = np.frombuffer(resident(os.path.join(w, "train_x.bin")), dtype=np.uint8)
    old_pages = n * d * 8 // 4096 - (40 << 20) // 4096
    old = np.nonzero(r[:old_pages])[0]
    runs = np.split(old, np.nonzero(np.diff(old) != 1)[0] + 1) if len(old) else []
    o = {
        "pace": os.environ.get("ENN_LOG_PACE", "1"),
        "n": n,
        "old_pages": int(old_pages),
        "old_resident_pages": len(old),
        "run_lengths": [len(x) for x in runs],
    }
    with open(out, "w") as f:
        json.dump(o, f)
    print(o)


if __name__ == "__main__":
    main()
