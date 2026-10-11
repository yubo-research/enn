from __future__ import annotations

import numpy as np

from enn.ennts import ENNTS, ENNTSConfig
from evals.stress_eval import format_larger, format_plain, format_smaller
from ops.qa import gaussian_nll


def make_series(length, dim, lookback, horizon, noise, rng):
    x = rng.normal(size=(length, dim))
    y = np.zeros((length, 1))
    for t in range(lookback - 1, length - horizon):
        window = x[t - lookback + 1 : t + 1, 0]
        y[t + horizon, 0] = np.tanh(float(window.mean()))
    y[:, 0] = y[:, 0] + rng.normal(scale=noise, size=length)
    return x, y


def score_time(model, xs, ys, t, lookback, horizon, min_obs):
    end = t - horizon
    if int(model.num_obs) < min_obs or end < lookback - 1:
        return [], [], []
    mus, ses, targets = [], [], []
    for series_id, x in enumerate(xs):
        window = x[end - lookback + 1 : end + 1]
        mu, se, _, _ = model.posterior(window, cutoff=end)
        mus.append(float(np.asarray(mu).ravel()[0]))
        ses.append(float(np.asarray(se).ravel()[0]))
        targets.append(float(ys[series_id][t, 0]))
    return mus, ses, targets


def add_time(model, xs, ys, t) -> None:
    stamp = np.asarray([t], dtype=np.int64)
    for series_id, x in enumerate(xs):
        model.add(series_id, x[t : t + 1], ys[series_id][t : t + 1], stamp)


def summarize(y, mu, se) -> tuple[float, float]:
    y_arr = np.asarray(y, dtype=float)
    mu_arr = np.asarray(mu, dtype=float)
    se_arr = np.asarray(se, dtype=float)
    if y_arr.size == 0:
        return float("nan"), float("nan")
    scale = float(np.std(y_arr))
    if scale == 0.0:
        scale = 1.0
    nrmse = float(np.sqrt(np.mean((mu_arr - y_arr) ** 2)) / scale)
    return float(-gaussian_nll(y_arr, mu_arr, se_arr)), nrmse


def cross_validated_scores(n_series, length, dim, seed, noise=0.05, min_obs=18):
    rng = np.random.default_rng(seed)
    lookback, horizon = 5, 1
    built = [
        make_series(length, dim, lookback, horizon, noise, rng) for _ in range(n_series)
    ]
    xs = [item[0] for item in built]
    ys = [item[1] for item in built]
    model = _cv_model(lookback, dim, seed, noise)
    mus, ses, targets = [], [], []
    for t in range(length):
        got_mu, got_se, got_y = score_time(model, xs, ys, t, lookback, horizon, min_obs)
        mus.extend(got_mu)
        ses.extend(got_se)
        targets.extend(got_y)
        add_time(model, xs, ys, t)
    return summarize(targets, mus, ses)


def _cv_model(lookback, dim, seed, noise) -> ENNTS:
    width = max(1.0, float(np.sqrt(lookback * dim)))
    return ENNTS(
        ENNTSConfig(
            lookback=lookback,
            num_tables=4,
            num_projections=3,
            bucket_width=width,
            k=6,
            probe=1,
            seed=int(seed),
            aleatoric_variance_scale=float(noise) ** 2,
        )
    )


def format_cv_line(dim, seed, loglik, nrmse) -> str:
    return (
        "EVAL: "
        f"{format_plain('d', dim)} "
        f"{format_plain('seed', seed)} "
        f"{format_larger('loglik', f'{loglik:.4f}')} "
        f"{format_smaller('nrmse', f'{nrmse:.4f}')}"
    )


def run_ennts_cv_eval() -> None:
    for dim, seed in ((2, 0), (2, 1), (4, 0), (4, 1)):
        loglik, nrmse = cross_validated_scores(3, 36, dim, seed)
        print(format_cv_line(dim, seed, loglik, nrmse), flush=True)
