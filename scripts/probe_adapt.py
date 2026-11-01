import json
import time
from pathlib import Path

import numpy as np

from enn import EpistemicNearestNeighbors
from enn.enn.enn_fit import enn_fit
from evals.harness.adapt import NUM_FIT_CANDIDATES, PARAMS, score
from evals.harness.adapt_problem import make_problem
from evals.harness.work_dirs import fresh_work_dir


def main():
    rows = []
    for seed in range(3, 6):
        problem = make_problem("xx", "misleading", seed, count=3128)
        model = EpistemicNearestNeighbors(
            problem.x[:0],
            problem.y[:0],
            input_transform=problem.input_transform,
            weighting=True,
            work_dir=fresh_work_dir(),
        )
        rng = np.random.default_rng(seed)
        for count in range(100, 3001, 100):
            start = time.perf_counter()
            token = model.add(
                problem.x[count - 100 : count], problem.y[count - 100 : count]
            )
            params = enn_fit(
                model,
                k=PARAMS.k_num_neighbors,
                num_fit_candidates=NUM_FIT_CANDIDATES,
                rng=rng,
                incremental=token,
            )
            rows.append(
                dict(
                    seed=seed,
                    n=count,
                    update_s=time.perf_counter() - start,
                    **score(
                        model.posterior(problem.x[3000:], params=params),
                        problem.y[3000:],
                    ),
                )
            )
    path = Path("_reports/adapt/saturation.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
