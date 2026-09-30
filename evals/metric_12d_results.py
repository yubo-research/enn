"""Per-seed checkpoint results of ``evals/metric_12d.py`` and their mean ± SE over seeds."""

from __future__ import annotations

from dataclasses import dataclass

from evals.stress_eval import format_larger, format_plain, format_smaller
from ops.stress import MeanSE, format_mean_se, mean_se


@dataclass(frozen=True)
class CheckpointResult:
    model: str
    num_obs: int
    loglik: float
    nrmse: float
    add_s: float
    query_s: float


@dataclass(frozen=True)
class CheckpointSummary:
    """One model and checkpoint aggregated over seeds."""

    model: str
    num_obs: int
    loglik: MeanSE
    nrmse: MeanSE
    add_s: MeanSE
    query_s: MeanSE


def format_seed_line(seed: int, result: CheckpointResult) -> str:
    """Per-seed progress line; deliberately not an EVAL line."""
    return (
        f"{format_plain('seed', seed)} "
        f"{format_plain('model', result.model)} "
        f"{format_plain('n', result.num_obs)} "
        f"{format_plain('loglik', f'{result.loglik:.4f}')} "
        f"{format_plain('nrmse', f'{result.nrmse:.4f}')} "
        f"{format_plain('add_s', f'{result.add_s:.4f}')} "
        f"{format_plain('query_s', f'{result.query_s:.4f}')}"
    )


def format_eval_line(summary: CheckpointSummary) -> str:
    def fmt(stat: MeanSE) -> str:
        return format_mean_se(stat, fmt=".4f")

    return (
        "EVAL: "
        f"{format_plain('model', summary.model)} "
        f"{format_plain('n', summary.num_obs)} "
        f"{format_larger('loglik', fmt(summary.loglik))} "
        f"{format_smaller('nrmse', fmt(summary.nrmse))} "
        f"{format_smaller('add_s', fmt(summary.add_s))} "
        f"{format_smaller('query_s', fmt(summary.query_s))}"
    )


def summarize(results: list[CheckpointResult]) -> list[CheckpointSummary]:
    """Mean ± SE over seeds per (model, n), in first-seen order."""
    groups: dict[tuple[str, int], list[CheckpointResult]] = {}
    for r in results:
        groups.setdefault((r.model, r.num_obs), []).append(r)
    return [
        CheckpointSummary(
            model=model,
            num_obs=num_obs,
            loglik=mean_se([r.loglik for r in rows]),
            nrmse=mean_se([r.nrmse for r in rows]),
            add_s=mean_se([r.add_s for r in rows]),
            query_s=mean_se([r.query_s for r in rows]),
        )
        for (model, num_obs), rows in groups.items()
    ]
