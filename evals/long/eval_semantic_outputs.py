"""Reproduce y-y experiments from the repository root.

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 PYTHONPATH=src:. python evals/long/eval_semantic_outputs.py
Pass --full for the requested 3 x 3 x 4 dimension/output/training-size grid.
The default is a repeated-seed pilot, with four cases and all four controls.
Memory records Python allocations per method and process peak RSS, including
native code. RSS is a cumulative high-water mark, not a per-method increment.
"""

import argparse
import csv
import resource
from itertools import product
from pathlib import Path

import numpy as np

from evals.long.semantic_output_problem import (
    method_embeddings,
    output_projection,
    synthetic_problem,
)
from evals.long.semantic_output_scoring import benchmark_method

ROOT = Path("_reports/semantic_outputs")
SCENARIOS = ("A_aligned", "B_independent", "C_opposing", "D_misleading")


def write_rows(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    """Uncertainty is across seeds, never across correlated output columns."""
    groups = {}
    for row in rows:
        key = tuple(
            row[name] for name in ("dimensions", "outputs", "n", "scenario", "method")
        )
        groups.setdefault(key, []).append(row)
    result = []
    for key, values in groups.items():
        record = dict(zip(("dimensions", "outputs", "n", "scenario", "method"), key))
        record["repetitions"] = len(values)
        for metric in (
            "nrmse",
            "loglik",
            "coverage95",
            "selection_seconds",
            "fit_seconds",
            "prediction_seconds",
            "python_peak_bytes",
        ):
            samples = np.array([row[metric] for row in values])
            record[metric + "_mean"] = samples.mean()
            record[metric + "_se"] = samples.std(ddof=1) / np.sqrt(len(samples))
        result.append(record)
    return result


def run_case(dimensions, outputs, n, seed, directory):
    projection = output_projection(outputs, directory / "cache")
    rows, individual = [], []
    splits = (n, n + max(30, n // 3), n + max(30, n // 3) + 200)
    for scenario in SCENARIOS:
        x, y = synthetic_problem(seed, dimensions, projection, scenario, splits[-1])
        for method, embeddings in method_embeddings(projection, seed).items():
            settings = (
                ([1], [0])
                if method == "independent"
                else (
                    [rank for rank in (1, 2, 4, 8) if rank < outputs],
                    [0, 0.1, 0.3, 1],
                )
            )
            record, metrics = benchmark_method(
                x, y, splits, (projection.names, embeddings), settings
            )
            record.update(
                dimensions=dimensions,
                outputs=outputs,
                n=n,
                seed=seed,
                scenario=scenario,
                method=method,
                process_peak_rss_bytes=resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss
                * 1024,
            )
            record.update({name: values.mean() for name, values in metrics.items()})
            rows.append(record)
            for output in range(outputs):
                item = {
                    name: record[name]
                    for name in (
                        "dimensions",
                        "outputs",
                        "n",
                        "seed",
                        "scenario",
                        "method",
                    )
                }
                item.update(
                    output=output,
                    **{name: values[output] for name, values in metrics.items()},
                )
                individual.append(item)
    return rows, individual


def evaluate(full=False, seeds=3, directory=ROOT):
    directory.mkdir(parents=True, exist_ok=True)
    grid = (
        product([20, 50, 100], [5, 20, 60], [50, 100, 200, 500])
        if full
        else product([20], [5], [50, 200])
    )
    rows, individual = [], []
    for dimensions, outputs, n in grid:
        for seed in range(seeds):
            new, per_output = run_case(dimensions, outputs, n, seed, directory)
            rows.extend(new)
            individual.extend(per_output)
            write_rows(directory / "raw.csv", rows)
            write_rows(directory / "per_output.csv", individual)
            print(f"D={dimensions}, outputs={outputs}, n={n}, seed={seed}", flush=True)
    write_rows(directory / "summary.csv", summarize(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--directory", type=Path, default=ROOT)
    arguments = parser.parse_args()
    if arguments.seeds < 2:
        parser.error("at least two seeds are required to report uncertainty")
    evaluate(arguments.full, arguments.seeds, arguments.directory)


if __name__ == "__main__":
    main()
