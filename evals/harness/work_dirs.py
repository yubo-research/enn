from __future__ import annotations

import atexit
import os
import shutil
import tempfile

_ROOT = tempfile.mkdtemp(prefix="enn_work_dirs_")
_OWNER_PID = os.getpid()


def _remove_root() -> None:
    if os.getpid() == _OWNER_PID:
        shutil.rmtree(_ROOT, ignore_errors=True)


atexit.register(_remove_root)


def fresh_work_dir() -> str:
    os.makedirs(_ROOT, exist_ok=True)
    return tempfile.mkdtemp(dir=_ROOT)
