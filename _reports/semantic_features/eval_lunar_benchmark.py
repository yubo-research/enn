"""Paired prediction and 30-repetition optimization benchmark for frozen-noise tlunar."""

import argparse
import json
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from _reports.semantic_features.tlunar import DESCRIPTIONS, NAMES, tlunar_fn
from enn import SemanticInputTransform, create_optimizer
from enn.turbo.config import (
    ENNIndexDriver,
    ENNSurrogateConfig,
    turbo_enn_config,
)
from evals.long.eval_semantic_benchmark import evaluate_prediction, write_csv

ROOT = Path(__file__).resolve().parent
METHODS = ("real", "semantic_only", "both")
COUNTS = (30, 100, 300, 1000)
BUDGET = 300
INITIAL = 30
BATCH = 10


def lunar_projection():
    path = ROOT / "lunar_projection.npz"
    if path.exists():
        transform = SemanticInputTransform.load(path)
        if transform.names != NAMES:
            raise ValueError(
                "cached lunar projection has historical parameter names; preserve "
                "historical artifacts and use a clean ROOT before rerunning"
            )
        return transform
    transform = SemanticInputTransform(NAMES, [[-1, 1]] * 12, DESCRIPTIONS)
    transform.save(path)
    (ROOT / "lunar_metadata.json").write_text(
        json.dumps(dict(zip(NAMES, DESCRIPTIONS)), indent=2) + "\n"
    )
    return transform


def returns(x):
    return np.array([tlunar_fn(row, num_denoise=1) for row in x]).reshape(-1, 1)


def prediction_case(seed):
    path = ROOT / f"lunar_prediction_seed{seed}.npz"
    if not path.exists():
        x = np.random.default_rng(seed).uniform(-1, 1, (1256, 12))
        np.savez(path, x=x, y=returns(x))
    with np.load(path) as data:
        x, y = data["x"], data["y"]
    z = lunar_projection().transform(x)
    features = {"real": z[:, :12], "semantic_only": z[:, 12:], "both": z}
    rows = []
    for count in COUNTS:
        for method, values in features.items():
            nrmse, loglik = evaluate_prediction(y, count, values)
            rows.append(
                {
                    "seed": seed,
                    "n": count,
                    "method": method,
                    "nrmse": nrmse,
                    "loglik": loglik,
                }
            )
    print(f"prediction seed {seed} complete", flush=True)
    return rows


def optimizer(method, seed, directory):
    transform = None if method == "real" else lunar_projection()
    surrogate = ENNSurrogateConfig(
        index_driver=ENNIndexDriver.BPANN_DISK,
        work_dir=directory,
        weighting=True,
        input_transform=transform,
        semantic_only=method == "semantic_only",
    )
    return create_optimizer(
        bounds=np.array([[-1, 1]] * 12, dtype=float),
        config=turbo_enn_config(enn=surrogate, num_init=INITIAL),
        rng=np.random.default_rng(seed),
    )


def optimization_case(job):
    method, seed = job
    path = ROOT / "lunar_runs" / f"{method}_{seed:02d}.npz"
    if path.exists():
        return str(path)
    with tempfile.TemporaryDirectory(prefix="enn_lunar_") as directory:
        opt = optimizer(method, seed, directory)
        xs, ys = [], []
        for count in range(0, BUDGET, BATCH):
            x = opt.ask(BATCH)
            y = returns(x)
            opt.tell(x, -y)
            xs.extend(x)
            ys.extend(y.ravel())
        del opt
    x, y = np.array(xs), np.array(ys)
    best = x[np.argmax(y)]
    validation = np.array(
        [tlunar_fn(best, num_denoise=1, noise_seed_0=s) for s in range(1, 11)]
    )
    path.parent.mkdir(exist_ok=True)
    np.savez(path, x=x, y=y, validation=validation)
    print(f"optimization {method}, seed {seed}: best {y.max():.1f}", flush=True)
    return str(path)


def optimization_rows(paths):
    rows = []
    for path in paths:
        method, seed = Path(path).stem.rsplit("_", 1)
        with np.load(path) as data:
            for count in (30, 100, 200, 300):
                rows.append(
                    {
                        "seed": int(seed),
                        "method": method,
                        "n": count,
                        "best_return": float(data["y"][:count].max()),
                        "validation_return": float(data["validation"].mean())
                        if count == BUDGET
                        else "",
                    }
                )
    return rows


def evaluate(*, workers=4, stage="all"):
    lunar_projection()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        if stage in ("prediction", "all"):
            rows = [r for case in pool.map(prediction_case, range(5)) for r in case]
            write_csv(ROOT / "lunar_prediction.csv", rows)
        if stage in ("optimization", "all"):
            jobs = [(m, s) for s in range(30) for m in METHODS]
            paths = list(pool.map(optimization_case, jobs))
            write_csv(ROOT / "lunar_optimization.csv", optimization_rows(paths))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--stage", choices=["prediction", "optimization", "all"], default="all"
    )
    args = parser.parse_args()
    evaluate(workers=args.workers, stage=args.stage)


if __name__ == "__main__":
    main()
