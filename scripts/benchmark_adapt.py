"""Record paired adaptation quality and steady-state update latency.

Run with PYTHONPATH=src:. python scripts/benchmark_adapt.py --output DIR.
Use --reference DIR with saved adaptive.py and adaptive_model.py for a paired comparison.
Run each version separately, on an otherwise idle machine.
Use --geometry-weight 30 to evaluate the opt-in fixed-weight update policy.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np


def load_reference(directory):
    import enn

    native = next(directory.glob("enn_rust*.so"), None)
    if native is not None:
        spec = importlib.util.spec_from_file_location("enn.enn_rust", native)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        enn.enn_rust = module
    for name in ("adaptive_model", "adaptive"):
        spec = importlib.util.spec_from_file_location(
            f"enn.{name}", directory / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        setattr(enn, name, module)
    enn.AdaptiveSemanticENN = module.AdaptiveSemanticENN


def measure(kind, case, seed, counts, geometry_weight):
    from enn import AdaptiveSemanticENN
    from enn.evals.adapt import score
    from enn.evals.adapt_problem import make_problem

    problem = make_problem(kind, case, seed, count=max(counts) + 128)
    model = AdaptiveSemanticENN(
        input_transform=problem.input_transform,
        output_projection=problem.output_projection,
    )
    rows, predictions = [], []
    for count in counts:
        cpu_start = time.process_time()
        start = time.perf_counter()
        model.add(
            problem.x[model.num_seen : count],
            problem.y[model.num_seen : count],
            geometry_weight=geometry_weight,
        )
        elapsed = time.perf_counter() - start
        cpu_elapsed = time.process_time() - cpu_start
        posterior = model.posterior(problem.x[-128:])
        rows.append(
            dict(
                kind=kind,
                case=case,
                seed=seed,
                n=count,
                update_s=elapsed,
                update_cpu_s=cpu_elapsed,
                choice=asdict(model.choice),
                **score(posterior, problem.y[-128:]),
            )
        )
        predictions.append(np.stack((posterior.mu, posterior.se)))
    return rows, np.stack(predictions)


def benchmark(seeds, geometry_weight):
    rows, predictions = [], {}
    for kind in ("xx", "xy", "yy"):
        cases = (
            ("aligned", "opposing", "independent")
            if kind == "yy"
            else ("aligned", "misleading")
        )
        for case in cases:
            for seed in range(seeds):
                result, prediction = measure(
                    kind, case, seed, (30, 100, 300), geometry_weight
                )
                rows.extend(result)
                predictions[f"{kind}_{case}_{seed}"] = prediction
    latency = []
    for kind, case in (("xx", "misleading"), ("xy", "misleading"), ("yy", "opposing")):
        for seed in range(3):
            result, _ = measure(
                kind, case, seed, tuple(range(100, 3001, 100)), geometry_weight
            )
            latency.extend(result)
    return rows, predictions, latency


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--geometry-weight", type=float)
    args = parser.parse_args()
    if args.reference is not None:
        load_reference(args.reference)
    rows, predictions, latency = benchmark(args.seeds, args.geometry_weight)
    source = (args.reference or Path("src/enn")) / "adaptive_model.py"
    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "geometry_weight": args.geometry_weight,
        "quality": rows,
        "latency": latency,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "threads": {
            name: os.getenv(name)
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
        },
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "native_sha256": hashlib.sha256(
            Path(sys.modules["enn.enn_rust"].__file__).read_bytes()
        ).hexdigest(),
    }
    (args.output / "measurements.json").write_text(json.dumps(payload, indent=2) + "\n")
    np.savez_compressed(args.output / "predictions.npz", **predictions)


if __name__ == "__main__":
    main()
