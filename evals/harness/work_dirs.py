from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from functools import cache


def _remove_root(root: str, owner_pid: int) -> None:
    if os.getpid() == owner_pid:
        shutil.rmtree(root, ignore_errors=True)


@cache
def _process_root(pid: int) -> str:
    root = tempfile.mkdtemp(prefix="enn_work_dirs_")
    atexit.register(_remove_root, root, pid)
    return root


def fresh_work_dir() -> str:
    return tempfile.mkdtemp(dir=_process_root(os.getpid()))
