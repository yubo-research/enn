from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NUM_OBS = 10_000_000
TS_NUM_OBS_SWEEP: tuple[int, ...] = (100, 1_000, 10_000)
TS_NUM_TEST = 100
TS_NUM_SEEDS = 100


def pythonpath_env() -> dict[str, str]:
    src = str(REPO_ROOT / "src")
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src if not existing else f"{src}{os.pathsep}{existing}"
    return env


def emit_eval_line(line: str) -> None:
    if line.startswith("n = "):
        print(f"EVAL: {line}", flush=True)
        return
    print(line, flush=True)


def emit_eval_ts_line(line: str, num_obs: int) -> None:
    if line.startswith("posterior"):
        print(f"EVAL: {num_obs} {line}", flush=True)
        return
    print(line, flush=True)


def stream_stress_stdout(
    proc: subprocess.Popen[str],
    emit: Callable[[str], None] | None = None,
) -> int:
    emit_line = emit_eval_line if emit is None else emit
    assert proc.stdout is not None
    for raw in proc.stdout:
        emit_line(raw.rstrip("\n"))
    return proc.wait()


def run_enn_stress_eval(*, index_type: str, num_obs: int, work_dir: str | None = None) -> None:
    """Run ENN stress and prefix metric rows with ``EVAL:``."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "ops" / "stress.py"),
        "enn",
        index_type,
        str(num_obs),
    ]
    if work_dir is not None:
        cmd.extend(["--work-dir", work_dir])
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        text=True,
        env=pythonpath_env(),
        cwd=str(REPO_ROOT),
    ) as proc:
        rc = stream_stress_stdout(proc)
    if rc != 0:
        raise SystemExit(rc)


def run_draw_stress_eval(
    *,
    num_obs: int,
    num_test: int = TS_NUM_TEST,
    num_seeds: int = TS_NUM_SEEDS,
) -> None:
    """Run draw stress and prefix ``posterior...`` rows with ``EVAL: NUM_OBS``."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "ops" / "stress.py"),
        "draw",
        str(num_obs),
        str(num_test),
        f"--num-seeds={num_seeds}",
    ]
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        text=True,
        env=pythonpath_env(),
        cwd=str(REPO_ROOT),
    ) as proc:
        rc = stream_stress_stdout(
            proc,
            emit=lambda line: emit_eval_ts_line(line, num_obs),
        )
    if rc != 0:
        raise SystemExit(rc)
