#!/usr/bin/env python
"""Compare time-series predictors on synthetic series.

Heavy imports stay inside the command body so ``--help`` returns before NumPy
or ENN are loaded.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _ensure_repo_on_path() -> None:
    root = str(_REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


@click.group()
def cli() -> None:
    """Compare time-series predictors on synthetic series."""


@cli.command("compare")
@click.option(
    "--function",
    "kind",
    type=click.Choice(["arma", "periodic", "nonlinear"]),
    required=True,
    help="Synthetic series family.",
)
@click.option("--seed", type=int, default=0, show_default=True, help="RNG seed.")
@click.option(
    "--out",
    type=click.Path(),
    default="_reports/ennts",
    show_default=True,
    help="Directory for JSON rows.",
)
def compare_cmd(kind: str, seed: int, out: str) -> None:
    """Compare ENNTS, ENNTS+METLRN, flat ENN, and BPANN_DISK+AUTO."""
    _ensure_repo_on_path()
    from ops.qa_ts_lib import observation_grid, run_sweep

    run_sweep(kind, observation_grid(), int(seed), out)


def main(argv: list[str] | None = None) -> None:
    try:
        cli.main(args=argv, prog_name="qa_ts.py", standalone_mode=False)
    except click.ClickException as exc:
        exc.show()
        raise SystemExit(exc.exit_code) from None


def __getattr__(name: str):
    """Library names stay importable; the import itself waits until first use."""
    _ensure_repo_on_path()
    import ops.qa_ts_lib as lib

    try:
        return getattr(lib, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None


if __name__ == "__main__":
    main()
