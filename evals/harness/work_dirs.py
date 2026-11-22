from __future__ import annotations

import atexit
import contextlib
import os
import re
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

DEFAULT_PREFIX = "enn_work_dirs_"
_OWNER_MARK = re.compile(r"_pid(\d+)-[A-Za-z0-9_]+$")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def owner_pid(name: str) -> int | None:
    match = _OWNER_MARK.search(name)
    return int(match.group(1)) if match else None


def sweep_stale_work_dirs(parent: str | Path | None = None) -> list[Path]:
    root = Path(parent) if parent is not None else Path(tempfile.gettempdir())
    removed: list[Path] = []
    with contextlib.suppress(OSError):
        for entry in root.iterdir():
            pid = owner_pid(entry.name)
            if pid is None or pid == os.getpid() or _pid_alive(pid):
                continue
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry, ignore_errors=True)
                removed.append(entry)
    return removed


def _make_owned_dir(prefix: str, parent: str | Path | None) -> str:
    sweep_stale_work_dirs(parent)
    return tempfile.mkdtemp(
        prefix=f"{prefix}pid{os.getpid()}-",
        dir=None if parent is None else str(parent),
    )


def _remove_owned_dir(path: str, owner: int) -> None:
    if os.getpid() == owner:
        shutil.rmtree(path, ignore_errors=True)


def fresh_work_dir(
    prefix: str = DEFAULT_PREFIX, parent: str | Path | None = None
) -> str:
    path = _make_owned_dir(prefix, parent)
    atexit.register(_remove_owned_dir, path, os.getpid())
    return path


@contextlib.contextmanager
def scoped_work_dir(
    prefix: str = DEFAULT_PREFIX, parent: str | Path | None = None
) -> Iterator[str]:
    path = _make_owned_dir(prefix, parent)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
