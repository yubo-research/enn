#!/usr/bin/env python

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path

import click

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_evaluate(name: str) -> Callable[[], None]:
    path = REPO_ROOT / "evals" / f"eval_{name}.py"
    if not path.is_file():
        raise click.ClickException(f"missing eval module: {path}")
    spec = importlib.util.spec_from_file_location(f"evals_eval_{name}", path)
    if spec is None or spec.loader is None:
        raise click.ClickException(f"cannot load eval module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    evaluate = getattr(module, "evaluate", None)
    if not callable(evaluate):
        raise click.ClickException(f"{path} has no callable evaluate()")
    return evaluate


@click.command()
@click.argument("name")
def main(name: str) -> None:
    """Run ``evals/eval_NAME.py::evaluate()``."""
    load_evaluate(name)()


if __name__ == "__main__":
    main()
