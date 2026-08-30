#!/usr/bin/env python

from __future__ import annotations

import importlib.util
import os
import resource
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import click

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALS_DIR = REPO_ROOT / "evals"
EVAL_PREFIX = "eval_"

# Eval runs are intentionally constrained for reproducible, machine-friendly timing.
DEFAULT_EVAL_NUM_THREADS = 1
DEFAULT_EVAL_MEMORY_LIMIT_BYTES = 3 * 1024**3  # 3 GiB
_EVAL_CHILD_ENV = "_ENNBO_EVAL_CHILD"
_EVAL_INLINE_ENV = "ENNBO_EVAL_INLINE"
_THREAD_ENV_KEYS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "ENNBO_OMP_NUM_THREADS",
)


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


def apply_eval_thread_limits(num_threads: int = DEFAULT_EVAL_NUM_THREADS) -> None:
    """Force BLAS/OpenMP/Rayon/Faiss thread caps for single-threaded eval runs."""
    if num_threads < 1:
        raise ValueError(f"num_threads must be >= 1, got {num_threads}")
    value = str(num_threads)
    for key in _THREAD_ENV_KEYS:
        os.environ[key] = value


def apply_eval_memory_limit(
    limit_bytes: int = DEFAULT_EVAL_MEMORY_LIMIT_BYTES,
) -> None:
    """Cap process address space (RLIMIT_AS) so evals cannot grow past ``limit_bytes``."""
    if limit_bytes <= 0:
        return
    if not hasattr(resource, "RLIMIT_AS"):
        raise click.ClickException("RLIMIT_AS is unavailable on this platform")
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    new_hard = limit_bytes if hard == resource.RLIM_INFINITY else min(hard, limit_bytes)
    new_soft = new_hard if soft == resource.RLIM_INFINITY else min(soft, new_hard)
    resource.setrlimit(resource.RLIMIT_AS, (new_soft, new_hard))


def prepare_eval_process(
    *,
    num_threads: int = DEFAULT_EVAL_NUM_THREADS,
    memory_limit_bytes: int = DEFAULT_EVAL_MEMORY_LIMIT_BYTES,
) -> None:
    """Apply single-thread env and memory cap inside the process that runs evaluate()."""
    apply_eval_thread_limits(num_threads)
    apply_eval_memory_limit(memory_limit_bytes)


def _pythonpath_env(base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    root = str(REPO_ROOT)
    src = str(REPO_ROOT / "src")
    existing = env.get("PYTHONPATH", "")
    parts = [p for p in existing.split(os.pathsep) if p]
    for path in (src, root):
        if path not in parts:
            parts.insert(0, path)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _thread_limit_env(num_threads: int = DEFAULT_EVAL_NUM_THREADS) -> dict[str, str]:
    value = str(num_threads)
    return {key: value for key in _THREAD_ENV_KEYS}


def run_evaluate_inline(name: str) -> None:
    """Run ``evaluate()`` in this process after applying resource limits."""
    prepare_eval_process()
    load_evaluate(name)()


def run_evaluate_subprocess(name: str) -> None:
    """Run the eval in a child process so memory limits do not stick to the parent."""
    env = _pythonpath_env()
    env.update(_thread_limit_env())
    env[_EVAL_CHILD_ENV] = "1"
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "run", name],
        cwd=str(REPO_ROOT),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.stdout:
        sys.stdout.write(completed.stdout)
        sys.stdout.flush()
    if completed.stderr:
        sys.stderr.write(completed.stderr)
        sys.stderr.flush()
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


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
    """Run ``evals/eval_NAME.py::evaluate()`` single-threaded with a 3 GiB memory cap."""
    inline = os.environ.get(_EVAL_INLINE_ENV) == "1"
    child = os.environ.get(_EVAL_CHILD_ENV) == "1"
    if inline or child:
        run_evaluate_inline(name)
        return
    run_evaluate_subprocess(name)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
