"""H5: is the pooled method's gap caused by retrieval (recall) or by a worse learned metric?

For metrics learned on raw-BPANN pools (batch refit, and one-pass online), report:
  - recall@k of a raw-coordinate pool of size K' against exact top-k under the learned metric,
  - LOO / test LL when the same learned metric is served with exact neighbors.
Usage: python recall.py OUT.json
"""

import json
import sys

import numpy as np

import stream
from iaml_core import knn, mean_ll

rows = []
for name in ("sparse", "friedman", "aniso"):
    for seed in range(3):
        x, y, xt, yt = stream.make_data(name, seed)
        n = stream.N_TRAIN
        d = x.shape[1]
        learners = {"A_pool10": stream.Pooled(d, 10), "C_online10": stream.Pooled(d, 10, online=True)}
        for key, meth in learners.items():
            for m in range(stream.BATCH, n + 1, stream.BATCH):
                meth.update(x, y, m)
            a, c = meth.m.a, meth.m.c
            exact_loo = knn(x, x, stream.K, a, self_offset=0)
            exact_test = knn(xt, x, stream.K, a)
            rec = {}
            for mult in (1, 3, 10, 30):
                pool = meth.index.query(xt, stream.K * mult, exclude_self=False)
                rec[mult] = float(np.mean([len(set(p) & set(e)) / stream.K for p, e in zip(pool, exact_test)]))
            rows.append(
                dict(
                    problem=name,
                    seed=seed,
                    learner=key,
                    recall=rec,
                    loo_pooled=mean_ll(x, y, x, y, meth.loo_nbr(x, y, n), a, c),
                    loo_exact=mean_ll(x, y, x, y, exact_loo, a, c),
                    test_pooled=mean_ll(xt, yt, x, y, meth.test_nbr(x, n, xt), a, c),
                    test_exact=mean_ll(xt, yt, x, y, exact_test, a, c),
                )
            )
            print(json.dumps(rows[-1]), flush=True)
with open(sys.argv[1], "w") as fh:
    json.dump(rows, fh)
