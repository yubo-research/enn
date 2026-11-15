import os
import sys

os.environ.setdefault("RAYON_NUM_THREADS", "1")
import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams, PosteriorFlags


def main():
    D, N, W, OUT = int(sys.argv[1]), int(float(sys.argv[2])), sys.argv[3], sys.argv[4]
    rng = np.random.default_rng(7)
    x = rng.random((N, D))
    y = np.sin(3 * x[:, :5]).sum(1, keepdims=True) + 0.1 * rng.standard_normal((N, 1))
    m = EpistemicNearestNeighbors(x[:1], y[:1], work_dir=W)
    for i in range(1, N, 1000):
        m.add(x[i : i + 1000], y[i : i + 1000])
    q = rng.random((100, D))
    p = m.posterior(
        q,
        params=ENNParams(
            k_num_neighbors=10,
            epistemic_variance_scale=1.0,
            aleatoric_variance_scale=0.1,
        ),
        flags=PosteriorFlags(observation_noise=False),
    )
    np.save(OUT, np.concatenate([np.asarray(p.mu).ravel(), np.asarray(p.se).ravel()]))


if __name__ == "__main__":
    main()
