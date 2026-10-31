import json
import time
from pathlib import Path

from enn import AdaptiveSemanticENN
from enn.evals.adapt import score
from enn.evals.adapt_problem import make_problem


def main():
    rows = []
    for seed in range(3, 6):
        problem = make_problem("xx", "misleading", seed, count=3128)
        model = AdaptiveSemanticENN(input_transform=problem.input_transform)
        for count in range(100, 3001, 100):
            start = time.perf_counter()
            model.add(problem.x[count - 100 : count], problem.y[count - 100 : count])
            rows.append(
                dict(
                    seed=seed,
                    n=count,
                    update_s=time.perf_counter() - start,
                    retained=model.num_retained,
                    replay_bytes=sum(array.nbytes for array in model._sample),
                    **score(model.posterior(problem.x[3000:]), problem.y[3000:]),
                )
            )
    path = Path("_reports/adapt/saturation.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
