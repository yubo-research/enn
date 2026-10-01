import time

import numpy as np
import pandas as pd

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_class_support import enn_neighbor_distances_and_indices
from enn.enn.enn_params import ENNParams
from enn.turbo.config.enn_x_scaling import ENNScaleX


def benchmark_d_scaling(ds=[100, 1000, 5000, 10000], n=1000, num_candidates=5000):
    print(f"Benchmarking scaling with D (N={n}, num_candidates={num_candidates})\n")

    results = []

    for d in ds:
        print(f"Running D={d}...")
        row = {"D": d}


        rng = np.random.default_rng(0)
        train_x = rng.random((n, d))
        train_y = rng.random((n, 1))
        cand_x = rng.random((num_candidates, d))
        ENNParams(
            k_num_neighbors=10,
            epistemic_variance_scale=1.0,
            aleatoric_variance_scale=0.1,
        )


        t0 = time.perf_counter()
        model = EpistemicNearestNeighbors(train_x, train_y, scale_x=ENNScaleX.ON)
        row["ENN_Init (s)"] = time.perf_counter() - t0


        t0 = time.perf_counter()
        _, _ = enn_neighbor_distances_and_indices(
            model.rust_backend,
            cand_x,
            search_k=10,
            exclude_nearest=False,
        )
        row["ENN_Search (s)"] = time.perf_counter() - t0

        results.append(row)

    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False))


    print("\nEmpirical scaling (log-log slope vs D):")
    for col in df.columns[1:]:
        y = np.log(df[col].values[-2:])
        x = np.log(df["D"].values[-2:])
        slope = (y[1] - y[0]) / (x[1] - x[0])
        print(f"  {col:20}: {slope:.2f}")


if __name__ == "__main__":
    benchmark_d_scaling(ds=[100, 1000, 5000, 10000])
