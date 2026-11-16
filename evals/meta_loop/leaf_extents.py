import json
import sys
from pathlib import Path

import numpy as np

META_WORDS = 4 + 65


def load(store):
    p = Path(store)
    meta = (
        p / "index" / "tree_meta.bin" if (p / "index").exists() else p / "tree_meta.bin"
    )
    w = np.fromfile(meta, dtype=np.uint32)
    return w[: len(w) // META_WORDS * META_WORDS].reshape(-1, META_WORDS)


def analyze(store, gap=2):
    w = load(store)
    kind, ln = w[:, 0] >> 24, w[:, 0] & 0xFFFFFF
    leaves, runs, spans = [], [], []
    for pid in np.flatnonzero(kind == 2):
        kids = w[pid, 4 : 4 + ln[pid]]
        kids = kids[kids < len(w)]
        if len(kids) == 0 or not np.all(kind[kids] == 1):
            continue
        k = np.sort(kids.astype(np.int64))
        leaves.append(len(k))
        runs.append(1 + int(np.sum(np.diff(k) > gap)))
        spans.append(int(k[-1] - k[0] + 1))
    r = np.array(runs)
    return {
        "pages": int(np.sum(kind != 0)),
        "leaf_parents": len(r),
        "leaves_per_parent": float(np.mean(leaves)),
        "runs_mean": float(r.mean()),
        "runs_p90": float(np.percentile(r, 90)),
        "one_run_frac": float(np.mean(r == 1)),
        "span_mean": float(np.mean(spans)),
    }


if __name__ == "__main__":
    print(
        json.dumps(analyze(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2))
    )
