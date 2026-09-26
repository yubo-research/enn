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
EVAL_BUCKETS = ("long", "short")


def ensure_repo_on_sys_path() -> None:
    """So ``from evals...`` works when invoking ``./ops/evaluate.py`` directly."""
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _name_from_stem(stem: str) -> str:
    if not stem.startswith(EVAL_PREFIX):
        raise ValueError(f"eval module stem must start with {EVAL_PREFIX!r}: {stem}")
    return stem[len(EVAL_PREFIX) :]


def iter_eval_entries() -> list[tuple[str, Path]]:
    """Return ``(eval_id, path)`` for every ``evals/{bucket}/eval_*.py``, sorted by id."""
    entries: list[tuple[str, Path]] = []
    for bucket in EVAL_BUCKETS:
        bucket_dir = EVALS_DIR / bucket
        if not bucket_dir.is_dir():
            continue
        for path in sorted(bucket_dir.glob(f"{EVAL_PREFIX}*.py")):
            eval_id = f"{bucket}/{_name_from_stem(path.stem)}"
            entries.append((eval_id, path))
    entries.sort(key=lambda item: item[0])
    return entries


def list_eval_names(prefix: str | None = None) -> list[str]:
    """Return eval ids (``short/name``, ``long/name``), optionally filtered by prefix."""
    names = [eval_id for eval_id, _ in iter_eval_entries()]
    if prefix is None:
        return names
    return [name for name in names if name.startswith(prefix)]


def eval_module_path(eval_id: str) -> Path:
    """Resolve an eval id or unique bare name to its module path."""
    entries = dict(iter_eval_entries())
    if eval_id in entries:
        return entries[eval_id]

    # Bare name: unique match on the name segment after ``bucket/``.
    matches = [path for eid, path in entries.items() if eid.rsplit("/", 1)[-1] == eval_id]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise click.ClickException(f"ambiguous eval name {eval_id!r}; use short/... or long/...")
    raise click.ClickException(f"missing eval module: {eval_id}")


def load_evaluate(eval_id: str) -> Callable[[], None]:
    ensure_repo_on_sys_path()
    path = eval_module_path(eval_id)
    if not path.is_file():
        raise click.ClickException(f"missing eval module: {path}")
    spec = importlib.util.spec_from_file_location(f"evals_eval_{path.stem}", path)
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
    """Discover and run ``evals/{{short,long}}/eval_NAME.py`` modules."""


@cli.command("list")
@click.argument("prefix", required=False, default=None)
def list_cmd(prefix: str | None) -> None:
    """Print eval ids; optional PREFIX keeps ids that start with PREFIX."""
    for name in list_eval_names(prefix):
        click.echo(name)


@cli.command("run")
@click.argument("prefix", required=False, default=None)
def run_cmd(prefix: str | None) -> None:
    """Run ``evaluate()`` for every eval id selected by PREFIX (same filter as list)."""
    names = list_eval_names(prefix)
    if not names:
        label = "prefix" if prefix is not None else "selection"
        raise click.ClickException(f"no evals match {label}: {prefix!r}")
    for eval_id in names:
        load_evaluate(eval_id)()


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
