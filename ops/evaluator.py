#!/usr/bin/env python
"""Backward-compatible alias for ``ops.evaluate``."""

from __future__ import annotations

from ops.evaluate import (
    EVAL_BUCKETS,
    EVAL_PREFIX,
    EVALS_DIR,
    REPO_ROOT,
    cli,
    ensure_repo_on_sys_path,
    eval_module_path,
    iter_eval_entries,
    list_eval_names,
    load_evaluate,
    main,
)

__all__ = [
    "EVAL_BUCKETS",
    "EVAL_PREFIX",
    "EVALS_DIR",
    "REPO_ROOT",
    "cli",
    "ensure_repo_on_sys_path",
    "eval_module_path",
    "iter_eval_entries",
    "list_eval_names",
    "load_evaluate",
    "main",
]


if __name__ == "__main__":
    main()
