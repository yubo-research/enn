_CLI_DESCRIPTION = "Reproduce the y-y mechanism study; run from the repository root.\n\nOPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 PYTHONPATH=src:. python evals/long/eval_yy.py\n"

import argparse
import csv
import hashlib
import json
import platform
import resource
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np

from enn import SemanticMultiOutputENN
from enn.enn.enn_params import ENNParams
from evals.long.semantic_output_scoring import scores
from evals.long.yy_problem import (
    ALPHAS,
    CASES,
    DESCRIPTIONS,
    metadata,
    problem,
    projections,
)

METRICS = ("nrmse", "loglik", "coverage95", "clean_nrmse")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_model(projection, alpha, data, n):
    x, y, clean = data
    started = perf_counter()
    model = SemanticMultiOutputENN(
        projection, alpha=alpha, params=ENNParams(8, 1, 0), folds=3, seed=0
    ).fit(x[:n], y[:n])
    fit_seconds = perf_counter() - started
    started = perf_counter()
    posterior = model.posterior(x[n:])
    prediction_seconds = perf_counter() - started
    scale = y[:n].std(axis=0)
    metrics = scores(y[n:], posterior, scale)
    metrics["clean_nrmse"] = np.sqrt(
        np.mean(((posterior.mu[100:] - clean[n + 100 :]) / scale) ** 2, axis=0)
    )
    test = type(posterior)(posterior.mu[100:], posterior.se[100:])
    test_metrics = scores(y[n + 100 :], test, scale)
    test_metrics["clean_nrmse"] = metrics["clean_nrmse"]
    validation = np.mean(
        np.sqrt(np.mean(((posterior.mu[:100] - y[n : n + 100]) / scale) ** 2, axis=0))
    )
    return {
        **{name: float(values.mean()) for name, values in test_metrics.items()},
        "validation_nrmse": float(validation),
        "fit_seconds": fit_seconds,
        "prediction_seconds": prediction_seconds,
    }, test_metrics


def run_setting(projection_pair, n, seed, case, variance_cutoff):
    data = problem(seed, n, case)
    variants = projections(*projection_pair, data[1][:n], seed, case, variance_cutoff)
    base, individual = evaluate_model(variants["semantic"], 0, data, n)
    rows = [dict(method="independent", alpha=0, **base)]
    per_output = [
        dict(
            method="independent",
            alpha=0,
            output=j,
            **{key: values[j] for key, values in individual.items()},
        )
        for j in range(16)
    ]
    for method, projection in variants.items():
        candidates = [dict(method=method, alpha=0, **base)]
        for alpha in ALPHAS[1:]:
            record, individual = evaluate_model(projection, alpha, data, n)
            candidates.append(dict(method=method, alpha=alpha, **record))
            per_output.extend(
                dict(
                    method=method,
                    alpha=alpha,
                    output=j,
                    **{key: values[j] for key, values in individual.items()},
                )
                for j in range(16)
            )
        rows.extend(candidates)
        selected = min(candidates, key=lambda item: item["validation_nrmse"])
        rows.append(dict(selected, method=method + "_selected"))
    key = {"n": n, "seed": seed, "case": case, "variance_cutoff": variance_cutoff}
    return [dict(key, **row) for row in rows], [dict(key, **row) for row in per_output]


def environment(directory, args, elapsed):
    paths = list(directory.glob("*.npz")) + [
        Path(__file__),
        Path("evals/long/yy_problem.py"),
        Path("src/enn/semantic_multi_output.py"),
        Path("src/enn/semantic_outputs.py"),
    ]
    record = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "arguments": vars(args) | {"directory": str(args.directory)},
        "elapsed_seconds": elapsed,
        "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024,
        "descriptions": DESCRIPTIONS,
        "sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
        },
    }
    (directory / "environment.json").write_text(json.dumps(record, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=_CLI_DESCRIPTION)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--sizes", type=int, nargs="+", default=[30, 100, 300])
    parser.add_argument(
        "--variance-cutoffs", type=float, nargs="+", default=[0.3, 0.9, 1]
    )
    parser.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    parser.add_argument("--directory", type=Path, default=Path("_reports/yy"))
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    projection_pair = metadata(args.directory)
    rows, individual = [], []
    started = perf_counter()
    for variance_cutoff, n in product(args.variance_cutoffs, args.sizes):
        for seed in range(args.seeds):
            for case in args.cases:
                new, outputs = run_setting(
                    projection_pair, n, seed, case, variance_cutoff
                )
                rows.extend(new)
                individual.extend(outputs)
            write_csv(args.directory / "raw.csv", rows)
            print(
                f"variance_cutoff={variance_cutoff} n={n} seed={seed} elapsed={perf_counter() - started:.1f}s",
                flush=True,
            )
    write_csv(args.directory / "per_output.csv", individual)
    environment(args.directory, args, perf_counter() - started)


if __name__ == "__main__":
    main()
