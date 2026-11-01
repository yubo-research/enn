from __future__ import annotations

import atexit
import shutil
import tempfile

_ROOT = tempfile.mkdtemp(prefix="enn_work_dirs_")
atexit.register(shutil.rmtree, _ROOT, ignore_errors=True)


def fresh_work_dir() -> str:
    return tempfile.mkdtemp(dir=_ROOT)
