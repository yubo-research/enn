_CLI_DESCRIPTION = "Confirmatory seeds and ablations for report-adapt.pdf.\n\nRun from the repository root with PYTHONPATH=src:. python scripts/evaluate_adapt.py.\nShort evaluator defaults remain three seeds; this command uses 27 additional seeds.\n"

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

from enn.evals.adapt import run_adapt


def main():
    parser = argparse.ArgumentParser(description=_CLI_DESCRIPTION)
    parser.add_argument("--seeds", type=int, default=27)
    parser.add_argument("--seed-start", type=int, default=3)
    parser.add_argument(
        "--output", type=Path, default=Path("_reports/adapt/confirmatory")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    methods = (
        "raw",
        "fixed",
        "tuned_fixed",
        "adaptive",
        "adaptive_inputs",
        "adaptive_outputs",
        "adaptive_linear",
    )
    timings = {}
    for kind in ("xx", "xy", "yy"):
        start = time.perf_counter()
        used = methods if kind == "yy" else methods[:4] + ("adaptive_linear",)
        run_adapt(
            kind,
            seeds=args.seeds,
            seed_start=args.seed_start,
            methods=used,
            output=args.output / f"{kind}.json",
        )
        timings[kind] = time.perf_counter() - start
    files = [
        Path("src/enn") / name
        for name in (
            "adaptive.py",
            "adaptive_model.py",
            "evals/adapt.py",
            "evals/adapt_problem.py",
        )
    ]
    protocol = {
        "seeds": args.seeds,
        "seed_start": args.seed_start,
        "timings": timings,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "capacity": 300,
        "checkpoints": [30, 100, 300],
        "test_rows": 128,
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files
        },
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")


if __name__ == "__main__":
    main()
