from __future__ import annotations

from evals.scaling_n import run_eval


def evaluate() -> None:
    """BPANN_DISK + AUTO + OLS, 12-d stream to n=1e5, 3 seeds: resident (total, heap,
    memory-mapped), peak and disk memory (MiB) per checkpoint, and their log-log slopes in n."""
    run_eval("memory")
