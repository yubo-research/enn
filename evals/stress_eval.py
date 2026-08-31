from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NUM_OBS = 1_000_000
TS_NUM_OBS_SWEEP: tuple[int, ...] = (100, 1_000, 10_000)
TS_NUM_TEST = 100
TS_NUM_SEEDS = 100
TURBO_ENN_NUM_OBS = 1000
TURBO_ENN_NUM_ASK = 100
TURBO_ENN_Y_BEST_PREFIX = "y_best="

_SPACED_KV_RE = re.compile(r"(\w+)\s*=\s*(\S+)")
_EQ_KEY_RE = re.compile(r"(\w+)=")

_ENN_METRIC_DIR: dict[str, str] = {
    "query_s": "SMALLER",
    "segment_s": "SMALLER",
}
_TS_METRIC_DIR: dict[str, str] = {
    "avg_likelihood": "LARGER",
    "argmin_rms": "SMALLER",
    "argmin_hit_rate": "LARGER",
    "eval_s": "SMALLER",
}


def pythonpath_env() -> dict[str, str]:
    src = str(REPO_ROOT / "src")
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src if not existing else f"{src}{os.pathsep}{existing}"
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "RAYON_NUM_THREADS",
        "ENNBO_OMP_NUM_THREADS",
    ):
        env.setdefault(key, "1")
    return env


def format_plain(name: str, value: object) -> str:
    return f"{name} = {value}"


def format_larger(name: str, value: object) -> str:
    return f"LARGER({name}) = {value}"


def format_smaller(name: str, value: object) -> str:
    return f"SMALLER({name}) = {value}"


def format_directed(direction: str, name: str, value: object) -> str:
    if direction == "LARGER":
        return format_larger(name, value)
    if direction == "SMALLER":
        return format_smaller(name, value)
    return format_plain(name, value)


def _parse_spaced_kv(line: str) -> dict[str, str]:
    return {m.group(1): m.group(2) for m in _SPACED_KV_RE.finditer(line)}


def _parse_eq_kv_tail(tail: str) -> dict[str, str]:
    matches = list(_EQ_KEY_RE.finditer(tail))
    out: dict[str, str] = {}
    for i, match in enumerate(matches):
        key = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(tail)
        out[key] = tail[start:end].strip()
    return out


def format_enn_eval_line(line: str) -> str | None:
    """Rewrite an ENN stress metric row, or return None if it is not one."""
    if not line.startswith("n = "):
        return None
    kv = _parse_spaced_kv(line)
    parts: list[str] = []
    if "n" in kv:
        parts.append(format_plain("n", kv["n"]))
    for key, direction in _ENN_METRIC_DIR.items():
        if key in kv:
            parts.append(format_directed(direction, key, kv[key]))
    return "EVAL: " + " ".join(parts)


def format_ts_eval_line(line: str, num_obs: int) -> str | None:
    """Rewrite a draw-stress posterior row, or return None if it is not one."""
    if not line.startswith("posterior"):
        return None
    method, _, tail = line.partition(" ")
    kv = _parse_eq_kv_tail(tail)
    parts = [format_plain("n", num_obs), format_plain("method", method)]
    for key, direction in _TS_METRIC_DIR.items():
        if key in kv:
            parts.append(format_directed(direction, key, kv[key]))
    return "EVAL: " + " ".join(parts)


def format_turbo_enn_eval_line(time_seconds: float, y_best: float) -> str:
    return (
        "EVAL: "
        f"{format_smaller('time_seconds', time_seconds)} "
        f"{format_larger('y_best', y_best)}"
    )


def emit_eval_line(line: str) -> None:
    rewritten = format_enn_eval_line(line)
    if rewritten is not None:
        print(rewritten, flush=True)
        return
    print(line, flush=True)


def emit_eval_ts_line(line: str, num_obs: int) -> None:
    rewritten = format_ts_eval_line(line, num_obs)
    if rewritten is not None:
        print(rewritten, flush=True)
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
    index_type: str = "flat",
    work_dir: str | None = None,
) -> None:
    """Run draw stress and prefix ``posterior...`` rows with ``EVAL:``."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "ops" / "stress.py"),
        "draw",
        str(num_obs),
        str(num_test),
        f"--num-seeds={num_seeds}",
    ]
    if index_type != "flat":
        cmd.append(f"--index-type={index_type}")
    if work_dir is not None:
        cmd.extend(["--work-dir", work_dir])
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


def emit_turbo_enn_line(line: str, *, y_best_holder: list[float | None]) -> None:
    """Pass through turbo-enn rows; capture the trailing ``y_best=`` summary."""
    if line.startswith(TURBO_ENN_Y_BEST_PREFIX):
        y_best_holder[0] = float(line[len(TURBO_ENN_Y_BEST_PREFIX) :])
        return
    print(line, flush=True)


def run_turbo_enn_stress_eval(
    *,
    index_type: str = "flat",
    num_obs: int = TURBO_ENN_NUM_OBS,
    num_ask: int = TURBO_ENN_NUM_ASK,
    work_dir: str | None = None,
) -> None:
    """Run turbo-enn stress; end with ``EVAL: SMALLER(time_seconds) LARGER(y_best)``."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "ops" / "stress.py"),
        "turbo-enn",
        index_type,
        str(num_obs),
        str(num_ask),
    ]
    if work_dir is not None:
        cmd.extend(["--work-dir", work_dir])
    y_best_holder: list[float | None] = [None]
    t0 = time.perf_counter()
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        text=True,
        env=pythonpath_env(),
        cwd=str(REPO_ROOT),
    ) as proc:
        rc = stream_stress_stdout(
            proc,
            emit=lambda line: emit_turbo_enn_line(line, y_best_holder=y_best_holder),
        )
    running_time_seconds = time.perf_counter() - t0
    if rc != 0:
        raise SystemExit(rc)
    y_best = y_best_holder[0]
    if y_best is None:
        raise SystemExit("turbo-enn stress did not report y_best")
    print(format_turbo_enn_eval_line(running_time_seconds, y_best), flush=True)
