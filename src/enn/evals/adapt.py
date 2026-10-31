import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from enn import EpistemicNearestNeighbors, SemanticMultiOutputENN
from enn.enn.enn_params import ENNParams
from enn.evals.adapt_problem import make_problem

PARAMS = ENNParams(8, 1, 0)
CHECKPOINTS = (30, 100, 300)


def score(posterior, target):
    scale = np.maximum(target.std(axis=0), 1e-12)
    error = (posterior.mu - target) / scale
    variance = np.maximum((posterior.se / scale) ** 2, 1e-12)
    return {
        "coverage": float(np.mean(np.abs(error) <= 1.96 * np.sqrt(variance))),
        "nrmse": float(np.sqrt(np.mean(error**2))),
        "loglik": float(
            -0.5 * np.mean(np.log(2 * np.pi * variance) + error**2 / variance)
        ),
    }


def fixed_model(problem, count, raw=False):
    x, y = problem.x[:count], problem.y[:count]
    if problem.output_projection is not None and not raw:
        return SemanticMultiOutputENN(
            problem.output_projection, alpha=0.3, params=PARAMS, folds=3
        ).fit(x, y)
    return EpistemicNearestNeighbors(
        x, y, input_transform=None if raw else problem.input_transform
    )


def predict(model, x):
    if isinstance(model, EpistemicNearestNeighbors):
        return model.posterior(x, params=PARAMS)
    return model.posterior(x)


def run_case(kind, case, seed, methods):
    problem = make_problem(kind, case, seed)
    rows, models = [], {}
    for count in CHECKPOINTS:
        for method in methods:
            start = time.perf_counter()
            model = update_model(problem, count, method, models)
            update_s = time.perf_counter() - start
            start = time.perf_counter()
            posterior = predict(model, problem.x[300:])
            rows.append(
                dict(
                    kind=kind,
                    case=case,
                    seed=seed,
                    n=count,
                    method=method,
                    update_s=update_s,
                    query_s=time.perf_counter() - start,
                    choice=None if method in ("fixed", "raw") else asdict(model.choice),
                    **score(posterior, problem.y[300:]),
                )
            )
    return rows


def update_model(problem, count, method, models):
    if method in ("fixed", "raw"):
        return fixed_model(problem, count, raw=method == "raw")
    from enn.adaptive import AdaptiveSemanticENN

    if method not in models:
        models[method] = AdaptiveSemanticENN(
            input_transform=problem.input_transform,
            output_projection=problem.output_projection,
            linear_mean=method == "adaptive_linear",
            adaptation={
                "adaptive_inputs": "inputs",
                "adaptive_outputs": "outputs",
                "tuned_fixed": "none",
            }.get(method, "all"),
        )
    model = models[method]
    start = model.num_seen
    return model.add(problem.x[start:count], problem.y[start:count])


def run_adapt(
    kind, *, seeds=3, methods=("raw", "fixed", "adaptive"), output=None, seed_start=0
):
    cases = (
        ("aligned", "opposing", "independent")
        if kind == "yy"
        else ("aligned", "misleading")
    )
    rows = []
    for case in cases:
        for seed in range(seed_start, seed_start + seeds):
            rows.extend(run_case(kind, case, seed, methods))
    for case in cases:
        for method in methods:
            selected = [
                r
                for r in rows
                if r["case"] == case and r["method"] == method and r["n"] == 300
            ]
            values = np.array([r["nrmse"] for r in selected])
            print(
                f"ADAPT {kind} {case} {method} NRMSE={values.mean():.4f} SE={values.std(ddof=1) / np.sqrt(seeds):.4f}"
            )
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, indent=2) + "\n")
    return rows
