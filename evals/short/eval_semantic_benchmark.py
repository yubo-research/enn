"""Paired high-dimensional ENN regression benchmark; run from repository root.

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python ops/evaluate.py run short/semantic_benchmark
Install sentence-transformers to construct the cached projections on the first run.
"""

import csv
from pathlib import Path

import numpy as np

from enn import EpistemicNearestNeighbors, SemanticInputTransform
from enn.enn.enn_params import ENNParams

GROUPS = [
    "Temperature of the heating system in degrees Celsius",
    "Pressure of the hydraulic system in pascals",
    "Electrical voltage of the power supply in volts",
    "Speed of the rotating motor in revolutions per minute",
    "Concentration of dissolved salt in the water",
    "Brightness of the lighting system in lumens",
    "Length of the mechanical support in meters",
    "Duration of the processing stage in seconds",
]
ROOT = Path(__file__).resolve().parent


def projection_for(dimensions):
    path = ROOT / f"semantic_projection_d{dimensions}.npz"
    if path.exists():
        return SemanticInputTransform.load(path)
    transform = SemanticInputTransform(
        [f"variable_{i}" for i in range(dimensions)],
        [[0, 1]] * dimensions,
        [GROUPS[i % 8] for i in range(dimensions)],
    )
    transform.save(path)
    return transform


def objective(x, weights):
    latent = (x - 0.5) @ weights
    return (
        np.sin(latent[:, 0]) + 0.6 * latent[:, 1] + 0.3 * np.cos(latent[:, 2])
    ).reshape(-1, 1)


def problem(seed, dimensions):
    rng = np.random.default_rng(seed)
    x = rng.uniform(size=(1256, dimensions))
    group_weights = rng.normal(size=(8, 3))
    group_weights -= group_weights.mean(axis=0)
    aligned = group_weights[np.arange(dimensions) % 8]
    unrelated = rng.normal(size=(dimensions, 3))
    scale = np.sqrt(dimensions / 12)
    return x, aligned / scale, unrelated / scale, rng.permutation(dimensions)


def evaluate_prediction(y, train_count, features):
    z = features
    model = EpistemicNearestNeighbors(z[:train_count], y[:train_count])
    post = model.posterior(z[1000:], params=ENNParams(8, 1.0, 0.0))
    residual = (post.mu - y[1000:]).ravel()
    std = y[1000:].std()
    variance = np.maximum(post.se.ravel() ** 2, 1e-12)
    rmse = np.sqrt(np.mean(residual**2)) / std
    loglik = np.mean(
        -0.5 * (np.log(2 * np.pi * variance / std**2) + residual**2 / variance)
    )
    return float(rmse), float(loglik)


def run_case(seed, dimensions):
    transform = projection_for(dimensions)
    x, aligned, unrelated, permutation = problem(seed, dimensions)
    features = {
        "real": x - 0.5,
        "semantic_only": transform.transform(x)[:, dimensions:],
        "both": transform.transform(x),
    }
    shuffled = transform.transform(x[:, permutation])[:, dimensions:]
    features["shuffled"] = np.concatenate((x - 0.5, shuffled), axis=1)
    rows = []
    for scenario, weights in [
        ("A_related", aligned),
        ("B_independent", unrelated),
        ("C_permuted", aligned[permutation]),
    ]:
        y = objective(x, weights)
        for count in [30, 100, 300, 1000]:
            for method, values in features.items():
                nrmse, loglik = evaluate_prediction(y, count, values)
                rows.append(
                    {
                        "seed": seed,
                        "dimensions": dimensions,
                        "scenario": scenario,
                        "n": count,
                        "method": method,
                        "nrmse": nrmse,
                        "loglik": loglik,
                    }
                )
    return rows


def summarize(rows):
    summaries = []
    keys = sorted({(r["dimensions"], r["scenario"], r["n"], r["method"]) for r in rows})
    for dimensions, scenario, count, method in keys:
        selected = [
            r
            for r in rows
            if (r["dimensions"], r["scenario"], r["n"], r["method"])
            == (dimensions, scenario, count, method)
        ]
        record = {
            "dimensions": dimensions,
            "scenario": scenario,
            "n": count,
            "method": method,
            "seeds": len(selected),
        }
        for metric in ["nrmse", "loglik"]:
            values = np.array([r[metric] for r in selected])
            record[f"{metric}_mean"] = values.mean()
            record[f"{metric}_se"] = values.std(ddof=1) / np.sqrt(len(values))
        summaries.append(record)
    return summaries


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate():
    rows = []
    for dimensions in [50, 100, 200]:
        for seed in range(5):
            rows.extend(run_case(seed, dimensions))
            print(f"completed D={dimensions}, seed={seed}", flush=True)
    write_csv(ROOT / "semantic_raw.csv", rows)
    write_csv(ROOT / "semantic_summary.csv", summarize(rows))


if __name__ == "__main__":
    evaluate()
