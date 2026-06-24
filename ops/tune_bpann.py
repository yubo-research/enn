#!/usr/bin/env python

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import click

ROOT = Path(__file__).resolve().parents[1]
STRESS = ROOT / "ops" / "stress.py"

BPANN_ENV_KEYS: tuple[str, ...] = (
    "BPANN_INDEX_COMPACT_ROWS_PER_FRAGMENT",
    "BPANN_INDEX_COMPACT_FRAGMENT_MAX",
    "BPANN_SEARCH_ROWS_PER_FRAGMENT",
    "BPANN_SEARCH_FRAGMENT_BUDGET_MAX",
    "BPANN_SMALL_FRAGMENT_MERGE_ROWS",
)


@dataclass(frozen=True)
class TuneResult:
    env: dict[str, str]
    query_s: float
    segment_s: float


def run_stress(*, num_obs: int, env: dict[str, str]) -> tuple[float, float]:
    work_dir = tempfile.mkdtemp(prefix="bpann_tune_")
    merged = {**os.environ, **env}
    cmd = [
        sys.executable,
        str(STRESS),
        "enn",
        "bpann_disk",
        str(num_obs),
        "--work-dir",
        work_dir,
    ]
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env={**merged, "PYTHONPATH": str(ROOT / "src")},
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    last = lines[-1].split()
    query_s = float(last[-2])
    segment_s = float(last[-1])
    return query_s, segment_s


def default_grid() -> list[dict[str, str]]:
    rows_per_frag = ("5000", "10000", "20000")
    frag_max = ("16", "32")
    search_budget = ("2", "3", "4")
    out: list[dict[str, str]] = []
    for rows in rows_per_frag:
        for mx in frag_max:
            for budget in search_budget:
                out.append(
                    {
                        "BPANN_INDEX_COMPACT_ROWS_PER_FRAGMENT": rows,
                        "BPANN_INDEX_COMPACT_FRAGMENT_MAX": mx,
                        "BPANN_SEARCH_FRAGMENT_BUDGET_MAX": budget,
                    }
                )
    return out


@click.command()
@click.option("--num-obs", type=int, default=100_000, show_default=True)
@click.option(
    "--baseline-query-s",
    type=float,
    default=0.083,
    show_default=True,
    help="Reference query_s to beat (2x faster target).",
)
@click.option(
    "--baseline-segment-s",
    type=float,
    default=127.0,
    show_default=True,
    help="Reference segment_s to beat (2x faster target).",
)
def main(num_obs: int, baseline_query_s: float, baseline_segment_s: float) -> None:
    """Grid-search BPANN_* env vars via ops/stress.py for this deployment."""
    target_query = baseline_query_s / 2.0
    target_segment = baseline_segment_s / 2.0
    best: TuneResult | None = None
    for env in default_grid():
        query_s, segment_s = run_stress(num_obs=num_obs, env=env)
        if query_s > target_query or segment_s > target_segment:
            continue
        if best is None or (query_s + segment_s) < (best.query_s + best.segment_s):
            best = TuneResult(env=env, query_s=query_s, segment_s=segment_s)
    if best is None:
        click.echo("No configuration met 2x targets on both query_s and segment_s.")
        raise SystemExit(1)
    click.echo("Recommended BPANN env for this host:")
    for key in BPANN_ENV_KEYS:
        if key in best.env:
            click.echo(f"export {key}={best.env[key]}")
    click.echo(
        f"checkpoint query_s={best.query_s:.3f} segment_s={best.segment_s:.3f} "
        f"(targets query<={target_query:.3f} segment<={target_segment:.3f})"
    )


if __name__ == "__main__":
    main()
