_CLI_DESCRIPTION = "Record adaptation quality and steady-state update latency.\n\nRun with PYTHONPATH=src:. python scripts/benchmark_adapt.py --output DIR.\nRun on an otherwise idle machine.\n"

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np


def measure(kind, case, seed, counts):
    from enn.enn.enn_fitter import ENNStatefulFitter
    from evals.harness.adapt import (
        NUM_FIT_CANDIDATES,
        PARAMS,
        WORKFLOWS,
        alpha_summary,
        score,
        workflow_model,
    )
    from evals.harness.adapt_problem import make_problem

    problem = make_problem(kind, case, seed, count=max(counts) + 128)
    model = workflow_model(problem, WORKFLOWS["adaptive"])
    fitter = ENNStatefulFitter(
        k=PARAMS.k_num_neighbors,
        seed=int(np.random.default_rng(seed).integers(2**63 - 1)),
        input_transform=model.input_transform,
    )
    rows, predictions = [], []
    for count in counts:
        cpu_start = time.process_time()
        start = time.perf_counter()
        first = len(model)
        x, y = problem.x[first:count], problem.y[first:count]
        model.add(x, y)
        fitter.tell(x, y)
        params = fitter.ask(model, num_fit_candidates=NUM_FIT_CANDIDATES)
        elapsed = time.perf_counter() - start
        cpu_elapsed = time.process_time() - cpu_start
        posterior = model.posterior(problem.x[-128:], params=params)
        rows.append(
            dict(
                kind=kind,
                case=case,
                seed=seed,
                n=count,
                update_s=elapsed,
                update_cpu_s=cpu_elapsed,
                mean_alpha=alpha_summary(params),
                **score(posterior, problem.y[-128:]),
            )
        )
        predictions.append(np.stack((posterior.mu, posterior.se)))
    return rows, np.stack(predictions)


def benchmark(seeds):
    rows, predictions = [], {}
    for kind in ("xx", "xy", "yy"):
        cases = (
            ("aligned", "opposing", "independent")
            if kind == "yy"
            else ("aligned", "misleading")
        )
        for case in cases:
            for seed in range(seeds):
                result, prediction = measure(kind, case, seed, (30, 100, 300))
                rows.extend(result)
                predictions[f"{kind}_{case}_{seed}"] = prediction
    latency = []
    for kind, case in (("xx", "misleading"), ("xy", "misleading"), ("yy", "opposing")):
        for seed in range(3):
            result, _ = measure(kind, case, seed, tuple(range(100, 3001, 100)))
            latency.extend(result)
    return rows, predictions, latency


def main():
    parser = argparse.ArgumentParser(description=_CLI_DESCRIPTION)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()
    rows, predictions, latency = benchmark(args.seeds)
    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "quality": rows,
        "latency": latency,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "threads": {
            name: os.getenv(name)
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
        },
        "native_sha256": hashlib.sha256(
            Path(sys.modules["enn.enn_rust"].__file__).read_bytes()
        ).hexdigest(),
    }
    (args.output / "measurements.json").write_text(json.dumps(payload, indent=2) + "\n")
    np.savez_compressed(args.output / "predictions.npz", **predictions)


if __name__ == "__main__":
    main()
