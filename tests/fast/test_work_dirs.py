from __future__ import annotations

import gc
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors
from evals.harness.work_dirs import (
    _remove_owned_dir,
    fresh_work_dir,
    owner_pid,
    scoped_work_dir,
    sweep_stale_work_dirs,
)

ROOT = Path(__file__).resolve().parents[2]


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_scoped_work_dir_is_owner_tagged_and_removed_on_exit(tmp_path) -> None:
    with scoped_work_dir("enn_t_", parent=tmp_path) as work_dir:
        assert Path(work_dir).is_dir()
        assert owner_pid(Path(work_dir).name) == os.getpid()
    assert not Path(work_dir).exists()


def test_fresh_work_dir_removal_only_runs_in_its_owner(tmp_path) -> None:
    work_dir = fresh_work_dir("enn_t_", parent=tmp_path)
    _remove_owned_dir(work_dir, os.getpid() + 1)
    assert Path(work_dir).is_dir()
    _remove_owned_dir(work_dir, os.getpid())
    assert not Path(work_dir).exists()


def test_sweep_removes_only_dirs_of_dead_owners(tmp_path) -> None:
    dead = tmp_path / f"enn_t_pid{_dead_pid()}-abc_123"
    live = tmp_path / f"enn_t_pid{os.getppid()}-abc_123"
    untagged = tmp_path / "enn_t_abc_123"
    for path in (dead, live, untagged):
        path.mkdir()
    assert sweep_stale_work_dirs(tmp_path) == [dead]
    assert live.is_dir() and untagged.is_dir()


def test_dir_of_a_killed_process_is_swept_by_the_next_one(tmp_path) -> None:
    code = (
        "import os, signal, sys; sys.path.insert(0, sys.argv[1]); "
        "from evals.harness.work_dirs import fresh_work_dir; "
        "print(fresh_work_dir('enn_t_', parent=sys.argv[2]), flush=True); "
        "os.kill(os.getpid(), signal.SIGKILL)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, str(ROOT), str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == -signal.SIGKILL
    leaked = Path(proc.stdout.strip())
    assert leaked.is_dir()
    with scoped_work_dir("enn_t_", parent=tmp_path):
        assert not leaked.exists()


def test_model_dropped_after_its_dir_is_removed_does_not_recreate_it(tmp_path) -> None:
    rng = np.random.default_rng(0)
    work_dir = tmp_path / "store"
    model = EpistemicNearestNeighbors(
        rng.uniform(size=(50, 3)), rng.normal(size=(50, 1)), work_dir=str(work_dir)
    )
    shutil.rmtree(work_dir)
    del model
    gc.collect()
    assert not work_dir.exists()
