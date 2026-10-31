import tracemalloc
from time import perf_counter

import numpy as np

from enn import SemanticMultiOutputENN, SemanticOutputProjection


def scores(y, posterior, scale):
    errors = (posterior.mu - y) / scale
    variance = np.maximum((posterior.se / scale) ** 2, 1e-24)
    return {
        "nrmse": np.sqrt(np.mean(errors**2, axis=0)),
        "loglik": np.mean(
            -0.5 * (np.log(2 * np.pi * variance) + errors**2 / variance), axis=0
        ),
        "coverage95": np.mean(np.abs(errors) <= 1.96 * np.sqrt(variance), axis=0),
    }


def select_model(x, y, validation, definition, settings):
    best, best_score = None, np.inf
    names, embeddings = definition
    cutoffs, alphas = settings
    scale = np.maximum(y.std(axis=0), 1e-12)
    for cutoff in cutoffs:
        projection = SemanticOutputProjection.from_embeddings(
            names, embeddings, variance_cutoff=cutoff
        )
        for alpha in alphas:
            if alpha == 0 and cutoff != cutoffs[0]:
                continue
            model = SemanticMultiOutputENN(projection, alpha=alpha, folds=3).fit(x, y)
            score = scores(validation[1], model.posterior(validation[0]), scale)[
                "nrmse"
            ].mean()
            if score < best_score:
                best, best_score = model, score
    return best, best_score


def benchmark_method(x, y, splits, definition, settings):
    train, valid, test = splits
    started = perf_counter()
    model, validation_score = select_model(
        x[:train],
        y[:train],
        (x[train:valid], y[train:valid]),
        definition,
        settings,
    )
    selection_seconds = perf_counter() - started
    started = perf_counter()
    model.fit(x[:valid], y[:valid])
    fit_seconds = perf_counter() - started
    started = perf_counter()
    posterior = model.posterior(x[valid:test])
    prediction_seconds = perf_counter() - started
    peak = _memory_peak(model, x[:valid], y[:valid], x[valid:test])
    metrics = scores(y[valid:test], posterior, np.maximum(y[:valid].std(axis=0), 1e-12))
    record = {
        "rank": model.projection.loadings.shape[1],
        "alpha": model.alpha,
        "validation_nrmse": validation_score,
        "selection_seconds": selection_seconds,
        "fit_seconds": fit_seconds,
        "prediction_seconds": prediction_seconds,
        "python_peak_bytes": peak,
    }
    return record, metrics


def _memory_peak(model, x, y, query):
    tracemalloc.start()
    try:
        model.fit(x, y).posterior(query)
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
