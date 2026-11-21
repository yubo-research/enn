from __future__ import annotations

import atexit
import os
import shutil
import tempfile


def _remove_owned_dir(path: str, owner_pid: int) -> None:
    if os.getpid() == owner_pid:
        shutil.rmtree(path, ignore_errors=True)


def fresh_work_dir() -> str:
    path = tempfile.mkdtemp(prefix="enn_work_dirs_")
    atexit.register(_remove_owned_dir, path, os.getpid())
    return path
