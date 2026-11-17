import numpy as np
from scipy.cluster.vq import kmeans2


def two_means(P, rng):
    sub = P[rng.choice(len(P), min(len(P), 4000), replace=False)]
    cen, _ = kmeans2(sub, 2, minit="++", seed=rng, iter=10)
    left = ((P - cen[0]) ** 2).sum(1) <= ((P - cen[1]) ** 2).sum(1)
    if left.all() or not left.any():
        left = np.arange(len(P)) < len(P) // 2
    return left
