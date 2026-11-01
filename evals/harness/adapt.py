import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from enn import EpistemicNearestNeighbors
from enn.enn.enn_fitter import ENNStatefulFitter
from enn.enn.enn_params import ENNParams
from evals.harness.adapt_problem import make_problem
from evals.harness.work_dirs import fresh_work_dir

PARAMS = ENNParams(8, 1, 0)
CHECKPOINTS = (30, 100, 300)
PRIOR_ALPHA = 0.3
NUM_FIT_CANDIDATES = 30


@dataclass(frozen=True)
class Workflow:
    weighting: bool
    learned_outputs: bool = True
    output_alpha: float | None = None


WORKFLOWS = {
    "adaptive": Workflow(weighting=True),
    "adaptive_inputs": Workflow(weighting=True, learned_outputs=False),
    "adaptive_outputs": Workflow(weighting=False),
    "tuned_fixed": Workflow(weighting=False, output_alpha=PRIOR_ALPHA),
}


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
    if problem.output_projection is None or raw:
        model = EpistemicNearestNeighbors(
            x,
            y,
            input_transform=None if raw else problem.input_transform,
            work_dir=fresh_work_dir(),
        )
        return model, PARAMS
    model = EpistemicNearestNeighbors(
        x[:0],
        y[:0],
        output_projection=problem.output_projection,
        output_alpha=PRIOR_ALPHA,
        work_dir=fresh_work_dir(),
    )
    model.add(x, y)
    fitter = ENNStatefulFitter(k=PARAMS.k_num_neighbors, rng=np.random.default_rng(0))
    fitter.tell(x, y)
    params = fitter.ask(model, num_fit_candidates=1)
    return model, replace(
        PARAMS,
        output_blend=params.output_blend,
    )


def alpha_summary(params):
    blend = params.output_blend
    return None if blend is None else float(blend.alpha.mean())


def run_case(kind, case, seed, methods):
    problem = make_problem(kind, case, seed)
    rows, models = [], {}
    for count in CHECKPOINTS:
        for method in methods:
            start = time.perf_counter()
            model, params = update_model(problem, count, method, models, seed)
            update_s = time.perf_counter() - start
            start = time.perf_counter()
            posterior = model.posterior(problem.x[300:], params=params)
            rows.append(
                dict(
                    kind=kind,
                    case=case,
                    seed=seed,
                    n=count,
                    method=method,
                    update_s=update_s,
                    query_s=time.perf_counter() - start,
                    mean_alpha=alpha_summary(params),
                    **score(posterior, problem.y[300:]),
                )
            )
    return rows


def workflow_model(problem, workflow):
    return EpistemicNearestNeighbors(
        problem.x[:0],
        problem.y[:0],
        input_transform=problem.input_transform,
        output_projection=problem.output_projection
        if workflow.learned_outputs
        else None,
        output_alpha=workflow.output_alpha if problem.output_projection else None,
        weighting=workflow.weighting,
        work_dir=fresh_work_dir(),
    )


def update_model(problem, count, method, models, seed):
    if method in ("fixed", "raw"):
        return fixed_model(problem, count, raw=method == "raw")
    if method not in models:
        model = workflow_model(problem, WORKFLOWS[method])
        fitter = ENNStatefulFitter(
            k=PARAMS.k_num_neighbors,
            rng=np.random.default_rng(seed),
            input_transform=model.input_transform,
        )
        models[method] = (model, fitter)
    model, fitter = models[method]
    start = len(model)
    x, y = problem.x[start:count], problem.y[start:count]
    model.add(x, y)
    fitter.tell(x, y)
    return model, fitter.ask(model, num_fit_candidates=NUM_FIT_CANDIDATES)


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
