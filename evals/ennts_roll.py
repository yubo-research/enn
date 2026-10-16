from __future__ import annotations

import time

import numpy as np

from evals.ennts_cv import summarize
from evals.stress_eval import format_larger, format_plain, format_smaller


def roll_spec(length, cadence, lookback, horizon, burn, separate_y):
    stamps = np.arange(length, dtype=np.int64) * int(cadence)
    return {
        "stamps": stamps,
        "lookback": int(lookback),
        "horizon": int(horizon),
        "burn": int(burn),
        "separate_y": bool(separate_y),
    }


def build_series(make, n_series, seed):
    rng = np.random.default_rng(seed)
    built = [make(rng, series_id) for series_id in range(n_series)]
    xs = [item[0] for item in built]
    ys = [item[1] for item in built]
    return xs, ys


def query_step(model, xs, ys, t, spec):
    lookback = spec["lookback"]
    end = t - spec["horizon"]
    if t < spec["burn"] or end < lookback - 1:
        return [], [], [], 0.0
    windows = np.stack([x[end - lookback + 1 : end + 1] for x in xs])
    started = time.perf_counter()
    mu, se, _, _ = model.posterior(windows, cutoff=int(spec["stamps"][t - 1]))
    elapsed = time.perf_counter() - started
    mus = np.asarray(mu, dtype=float).ravel().tolist()
    ses = np.asarray(se, dtype=float).ravel().tolist()
    targets = [float(y[t, 0]) for y in ys]
    return mus, ses, targets, elapsed


def add_step(model, xs, ys, t, spec):
    stamp = spec["stamps"][t : t + 1]
    started = time.perf_counter()
    for sid, (x, y) in enumerate(zip(xs, ys, strict=True)):
        row_x = x[t : t + 1]
        row_y = y[t : t + 1]
        if spec["separate_y"]:
            model.add(sid, row_x, None, stamp)
            model.add_y(sid, stamp, row_y)
        else:
            model.add(sid, row_x, row_y, stamp)
    return time.perf_counter() - started


def rolling(model, xs, ys, spec):
    mus, ses, targets = [], [], []
    add_s = 0.0
    query_s = 0.0
    for t in range(len(spec["stamps"])):
        got_mu, got_se, got_y, elapsed = query_step(model, xs, ys, t, spec)
        mus.extend(got_mu)
        ses.extend(got_se)
        targets.extend(got_y)
        query_s += elapsed
        add_s += add_step(model, xs, ys, t, spec)
    loglik, nrmse = summarize(targets, mus, ses)
    return {
        "loglik": loglik,
        "nrmse": nrmse,
        "add_s": add_s,
        "query_s": query_s,
        "n": len(targets),
        "windows": int(model.num_windows),
    }


def format_stress_line(aspect, seed, scores, extra):
    body = " ".join(
        [
            format_plain("aspect", aspect),
            format_plain("seed", seed),
            format_plain("n", scores["n"]),
            format_larger("loglik", f"{scores['loglik']:.4f}"),
            format_smaller("nrmse", f"{scores['nrmse']:.4f}"),
            format_smaller("add_s", f"{scores['add_s']:.6f}"),
            format_smaller("query_s", f"{scores['query_s']:.6f}"),
            *extra,
        ]
    )
    return f"EVAL: {body}"
