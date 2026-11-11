import json
import os
import sys
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", "1")
import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams


def rss():
    d = {}
    for l in Path("/proc/self/status").read_text().splitlines():
        if l.startswith(("RssAnon", "RssFile")):
            d[l.split(":")[0]] = int(l.split()[1]) // 1024
    return d


out = {"import": rss()}
P = ENNParams(
    k_num_neighbors=10, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.1
)
m = EpistemicNearestNeighbors.open(sys.argv[1])
out["open"] = rss()
qr = np.random.default_rng(999)
for i in range(int(sys.argv[2])):
    m.posterior(qr.standard_normal((1, 1000)), params=P)
out["queries"] = rss()
print(json.dumps(out))
