#!/usr/bin/env python

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path

import click

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALS_DIR = REPO_ROOT / "evals"
EVAL_PREFIX = "eval_"


def ensure_repo_on_sys_path() -> None:
    """So ``from evals...`` works when invoking ``./ops/evaluator.py`` directly."""
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def eval_module_path(name: str) -> Path:
    return EVALS_DIR / f"{EVAL_PREFIX}{name}.py"


def list_eval_names() -> list[str]:
    names: list[str] = []
    for path in sorted(EVALS_DIR.glob(f"{EVAL_PREFIX}*.py")):
        names.append(path.stem[len(EVAL_PREFIX) :])
    return names


def load_evaluate(name: str) -> Callable[[], None]:
    ensure_repo_on_sys_path()
    path = eval_module_path(name)
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


@click.group()
def cli() -> None:
    """Discover and run ``evals/eval_NAME.py`` modules."""


@cli.command("list")
def list_cmd() -> None:
    """Print the names of all available evals."""
    for name in list_eval_names():
        click.echo(name)


@cli.command("run")
@click.argument("name")
def run_cmd(name: str) -> None:
    """Run ``evals/eval_NAME.py::evaluate()``."""
    load_evaluate(name)()


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
