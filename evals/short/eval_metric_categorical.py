from __future__ import annotations

from evals.metric_categorical import run_eval


def evaluate() -> None:
    """One-hot categorical inputs, n <= 3000, 5 seeds: cat_only (six categorical variables) and
    mixed (four continuous inputs of ranges 0.01..100 plus three categorical variables);
    BPANN_DISK with and without scale_x / AUTO, with and without tied one-hot columns:
    mean±se loglik, nrmse, add_s, query_s per problem, model and checkpoint."""
    run_eval()
