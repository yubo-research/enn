from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager

import modal

from ops.evaluate import REPO_ROOT

REMOTE_ROOT = "/root"
TIMEOUT_S = 24 * 3600
_IGNORE = ["**/__pycache__", "**/*.pyc"]

EvalResult = tuple[str, int, str]
RemoteCall = Callable[[str], Awaitable[tuple[int, str]]]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgomp1", "libopenblas0", "libblas3", "liblapack3")
    .pip_install(
        "numpy==2.4.6",
        "scipy==1.17.1",
        "click==8.4.2",
        "nds==0.4.3",
    )
    .env({"PYTHONPATH": f"{REMOTE_ROOT}{os.pathsep}{REMOTE_ROOT}/src"})
    .add_local_dir(REPO_ROOT / "src", f"{REMOTE_ROOT}/src", ignore=_IGNORE)
    .add_local_dir(REPO_ROOT / "evals", f"{REMOTE_ROOT}/evals", ignore=_IGNORE)
    .add_local_dir(REPO_ROOT / "ops", f"{REMOTE_ROOT}/ops", ignore=_IGNORE)
)

app = modal.App("enn-evals", image=image, include_source=False)


def eval_command(eval_id: str) -> list[str]:
    """Run exactly one eval (``run`` would treat the id as a prefix)."""
    code = "import sys; from ops.evaluate import load_evaluate; load_evaluate(sys.argv[1])()"
    return [sys.executable, "-c", code, eval_id]


def run_eval_captured(eval_id: str) -> tuple[int, str]:
    """Run one eval in a subprocess; return its exit code and merged stdout/stderr."""
    proc = subprocess.run(
        eval_command(eval_id),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(REPO_ROOT),
        check=False,
    )
    return proc.returncode, proc.stdout


@app.function(cpu=4.0, memory=8192, timeout=TIMEOUT_S)
def run_eval_remote(eval_id: str) -> tuple[int, str]:
    return run_eval_captured(eval_id)


async def _call_one(call: RemoteCall, eval_id: str) -> EvalResult:
    try:
        rc, output = await call(eval_id)
    except Exception as exc:
        return eval_id, 1, f"{type(exc).__name__}: {exc}\n"
    return eval_id, rc, output


async def iter_completed(call: RemoteCall, names: list[str]) -> AsyncIterator[EvalResult]:
    """Start one call per eval; yield each result as soon as its call finishes."""
    for fut in asyncio.as_completed([_call_one(call, name) for name in names]):
        yield await fut


def format_result(eval_id: str, rc: int, output: str) -> str:
    status = "ok" if rc == 0 else f"FAILED rc={rc}"
    body = output if output.endswith("\n") or not output else output + "\n"
    return f"===== {eval_id} ({status}) =====\n{body}===== end {eval_id} ====="


async def run_all(
    names: list[str],
    echo: Callable[[str], None],
    call: RemoteCall,
    context: AbstractAsyncContextManager[object],
) -> list[str]:
    """Echo each eval's full output once it completes; return ids of failed evals."""
    failed: list[str] = []
    async with context:
        async for eval_id, rc, output in iter_completed(call, names):
            echo(format_result(eval_id, rc, output))
            if rc != 0:
                failed.append(eval_id)
    return failed


def run_on_modal(names: list[str], echo: Callable[[str], None]) -> list[str]:
    echo(f"Running {len(names)} eval(s) on Modal as separate function calls...")
    return asyncio.run(run_all(names, echo, run_eval_remote.remote.aio, app.run()))
