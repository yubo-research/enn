from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams
from enn.ennts import ENNTS, ENNTSConfig
from enn.turbo.config.enn_index_driver import ENNIndexDriver
from enn.turbo.config.enn_surrogate_config import ENNStorage
from enn.turbo.config.enn_x_scaling import ENNMetricLearning, ENNScaleX

DIM = 10
LOOKBACK = 5
HORIZON = 1
OBS_PER_TIME = 10
K_NEIGHBORS = 8
NOISE = 0.1
EPISTEMIC = 1.0
ALEATORIC = NOISE * NOISE
ARMA_BURN = 20
NL_BURN = 8
X_JITTER = 0.05
ARMA_SINE = 0.25
ARMA_PERIOD = 30.0
PERIOD = 24.0
TRAIN_NUMERATOR = 3
TRAIN_DENOMINATOR = 4
AUTO_BATCH = 256

_PHI = np.eye(DIM) * 0.55
_PHI[np.arange(DIM), (np.arange(DIM) + 1) % DIM] = 0.12
_THETA = np.eye(DIM) * 0.25


def observation_grid() -> list[int]:
    return [100, 300, 1000, 3000, 10000, 30000, 100000]


def timestep_counts(n_obs: int, n_times: int, rng: np.random.Generator) -> np.ndarray:
    level = np.empty(n_times, dtype=float)
    acc = 0.0
    shock = rng.normal(scale=0.35, size=n_times)
    for t in range(n_times):
        acc = 0.95 * acc + float(shock[t])
        level[t] = acc
    level -= np.max(level)
    np.exp(level, out=level)
    gap = max(LOOKBACK + HORIZON + 2, int(n_times * 0.08))
    level[gap : n_times - 1 : gap] = 0.0
    probs = level / float(level.sum())
    return rng.multinomial(int(n_obs), probs).astype(np.int64)


def occupy(counts: np.ndarray) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    active: list[tuple[int, int]] = []
    next_id = 0
    for t, raw in enumerate(counts.tolist()):
        need = int(raw)
        while len(active) > need:
            sid, start = active.pop()
            runs.append((sid, start, t - start))
        while len(active) < need:
            active.append((next_id, t))
            next_id += 1
    end = len(counts)
    for sid, start in active:
        runs.append((sid, start, end - start))
    return runs


def _arma_path(n_steps: int, rng: np.random.Generator) -> np.ndarray:
    eps_prev = np.zeros(DIM)
    x_prev = np.zeros(DIM)
    xs = np.empty((n_steps, DIM))
    shocks = rng.normal(size=(n_steps, DIM))
    for i in range(n_steps):
        eps = shocks[i]
        x_prev = _PHI @ x_prev + eps + _THETA @ eps_prev
        xs[i] = x_prev
        eps_prev = eps
    return xs


def simulate_arma(t0: int, length: int, rng: np.random.Generator, noise: float):
    xs = _arma_path(length + ARMA_BURN, rng)[ARMA_BURN:]
    times = int(t0) + np.arange(length)
    signal = xs.mean(axis=1) + ARMA_SINE * np.sin(2.0 * np.pi * times / ARMA_PERIOD)
    return xs, signal + rng.normal(scale=float(noise), size=length)


def simulate_periodic(t0: int, length: int, rng: np.random.Generator, noise: float):
    times = int(t0) + np.arange(length)
    phase = 2.0 * np.pi * times / PERIOD
    offsets = 0.35 * np.arange(DIM)
    xs = np.sin(phase[:, None] + offsets[None, :])
    xs = xs + X_JITTER * rng.normal(size=xs.shape)
    signal = np.sin(phase) + 0.5 * np.cos(2.0 * phase) + 0.2 * xs[:, 0]
    return xs, signal + rng.normal(scale=float(noise), size=length)


def nonlinear_signal(x_obs: np.ndarray, x_burn: np.ndarray) -> np.ndarray:
    src = np.concatenate([x_burn, x_obs], axis=0)
    burn = int(x_burn.shape[0])
    out = np.empty(x_obs.shape[0])
    for i in range(x_obs.shape[0]):
        j = burn + i
        mixed = src[j - 1, 0] * src[j - 2, 1] + src[j - 3, 2]
        out[i] = np.tanh(mixed) + np.sin(src[j - 4, 3] + src[j - 5, 4])
    return out


def simulate_nonlinear(t0: int, length: int, rng: np.random.Generator, noise: float):
    del t0
    total = int(length) + NL_BURN
    shocks = rng.normal(size=(total, DIM))
    src = np.empty((total, DIM))
    x_prev = np.zeros(DIM)
    for i in range(total):
        x_prev = 0.45 * x_prev + shocks[i]
        src[i] = x_prev
    xs = src[NL_BURN:]
    signal = nonlinear_signal(xs, src[:NL_BURN])
    return xs, signal + rng.normal(scale=float(noise), size=length)


SIMULATORS = {
    "arma": simulate_arma,
    "periodic": simulate_periodic,
    "nonlinear": simulate_nonlinear,
}


def flatten_windows(x: np.ndarray) -> np.ndarray:
    n_rows = int(x.shape[0]) - LOOKBACK - HORIZON + 1
    width = LOOKBACK * int(x.shape[1])
    if n_rows <= 0:
        return np.zeros((0, width))
    flats = np.empty((n_rows, width), dtype=float)
    for i in range(n_rows):
        flats[i] = x[i : i + LOOKBACK].reshape(-1)
    return flats


def _train_prefix(sid: int, t0: int, x: np.ndarray, y: np.ndarray, t_cut: int):
    n_keep = min(len(y), max(0, int(t_cut) - int(t0)))
    if n_keep == 0:
        return None
    return (int(sid), int(t0), x[:n_keep], y[:n_keep])


def _window_split(t0: int, x: np.ndarray, y: np.ndarray, t_cut: int):
    flats = flatten_windows(x)
    n_rows = int(flats.shape[0])
    if n_rows == 0:
        return None
    start = LOOKBACK + HORIZON - 1
    targets = y[start : start + n_rows]
    times = np.arange(int(t0) + start, int(t0) + start + n_rows)
    return flats, targets, times < int(t_cut)


def _append_split(
    bucket: dict, flats: np.ndarray, targets: np.ndarray, train: np.ndarray
):
    if np.any(train):
        bucket["train_x"].append(flats[train])
        bucket["train_y"].append(targets[train])
    hold = ~train
    if np.any(hold):
        shaped = flats[hold].reshape(-1, LOOKBACK, DIM)
        bucket["test_x"].append(shaped)
        bucket["test_y"].append(targets[hold])


def _consume_run(kind: str, run: tuple[int, int, int], t_cut: int, rng, noise: float):
    sid, t0, length = run
    draw = np.random.default_rng(int(rng.integers(1, 2**31 - 1)))
    x, y = SIMULATORS[kind](int(t0), int(length), draw, noise)
    series = _train_prefix(sid, t0, x, y, t_cut)
    split = _window_split(t0, x, y, t_cut)
    return series, split


def _new_bucket() -> dict:
    return {"train_x": [], "train_y": [], "test_x": [], "test_y": [], "series": []}


def _pack_runs(kind: str, runs: list, t_cut: int, rng, noise: float) -> dict:
    bucket = _new_bucket()
    for run in runs:
        series, split = _consume_run(kind, run, t_cut, rng, noise)
        if series is not None:
            bucket["series"].append(series)
        if split is not None:
            _append_split(bucket, split[0], split[1], split[2])
    return bucket


def _cat(parts: list, shape: tuple[int, ...]) -> np.ndarray:
    if not parts:
        return np.zeros(shape)
    return np.concatenate(parts, axis=0)


def assemble(kind: str, n_obs: int, seed: int, noise: float) -> dict:
    n_times = int(n_obs) // OBS_PER_TIME
    rng = np.random.default_rng(int(seed))
    counts = timestep_counts(int(n_obs), n_times, rng)
    runs = occupy(counts)
    t_cut = (TRAIN_NUMERATOR * n_times) // TRAIN_DENOMINATOR
    bucket = _pack_runs(kind, runs, t_cut, rng, noise)
    width = LOOKBACK * DIM
    return {
        "counts": counts,
        "t_cut": int(t_cut),
        "n_times": int(n_times),
        "train_x": _cat(bucket["train_x"], (0, width)),
        "train_y": _cat(bucket["train_y"], (0,)),
        "test_x": _cat(bucket["test_x"], (0, LOOKBACK, DIM)),
        "test_y": _cat(bucket["test_y"], (0,)),
        "series": bucket["series"],
    }


def loglik_and_nrmse(
    y: np.ndarray, mu: np.ndarray, se: np.ndarray
) -> tuple[float, float]:
    var = np.maximum(np.asarray(se, dtype=float) ** 2, 1e-12)
    resid = np.asarray(y, dtype=float) - np.asarray(mu, dtype=float)
    nll = 0.5 * np.mean(np.log(2.0 * np.pi * var) + resid**2 / var)
    scale = float(np.std(y))
    if scale == 0.0:
        scale = 1.0
    nrmse = float(np.sqrt(np.mean(resid**2)) / scale)
    return float(-nll), nrmse


def _ennts(metric_learning: str = "none") -> ENNTS:
    width = float(np.sqrt(LOOKBACK * DIM))
    return ENNTS(
        ENNTSConfig(
            lookback=LOOKBACK,
            num_tables=8,
            num_projections=4,
            bucket_width=width,
            k=K_NEIGHBORS,
            probe=1,
            seed=0,
            horizon=HORIZON,
            cadence=1,
            aleatoric_variance_scale=ALEATORIC,
            epistemic_variance_scale=EPISTEMIC,
            metric_learning=metric_learning,
        )
    )


def _fill_ennts(model: ENNTS, series: list) -> None:
    for sid, t0, x, y in series:
        stamps = np.arange(int(t0), int(t0) + len(y), dtype=np.int64)
        model.add(int(sid), x, np.asarray(y, dtype=float).reshape(-1, 1), stamps)
    model.sync_index()


def _timed(work):
    started = time.perf_counter()
    value = work()
    return value, time.perf_counter() - started


def predict_ennts(series: list, test_x: np.ndarray, t_cut: int):
    model = _ennts()
    _, add_s = _timed(lambda: _fill_ennts(model, series))
    packed, query_s = _timed(lambda: model.posterior(test_x, cutoff=int(t_cut)))
    mu, se, _, _ = packed
    return (
        np.asarray(mu, dtype=float).ravel(),
        np.asarray(se, dtype=float).ravel(),
        add_s,
        query_s,
    )


def predict_ennts_metlrn(series: list, test_x: np.ndarray, t_cut: int):
    model = _ennts("auto")
    _, add_s = _timed(lambda: _fill_ennts(model, series))
    packed, query_s = _timed(lambda: model.posterior(test_x, cutoff=int(t_cut)))
    mu, se, _, _ = packed
    weights = model.metric_weights
    return {
        "mu": np.asarray(mu, dtype=float).ravel(),
        "se": np.asarray(se, dtype=float).ravel(),
        "heldout_gain": model.heldout_gain,
        "weights": None if weights is None else np.asarray(weights, dtype=float),
        "morph_remaining": int(model.metric_morph_remaining),
        "uses_learned": bool(model.uses_learned_metric),
        "add_s": add_s,
        "query_s": query_s,
    }


def _flat_test(test_x: np.ndarray) -> np.ndarray:
    return np.asarray(test_x, dtype=float).reshape(test_x.shape[0], LOOKBACK * DIM)


def _posterior_pair(model, flat: np.ndarray):
    params = ENNParams(K_NEIGHBORS, EPISTEMIC, ALEATORIC)
    post = model.posterior(flat, params=params)
    return np.asarray(post.mu, dtype=float).ravel(), np.asarray(
        post.se, dtype=float
    ).ravel()


def predict_enn(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray):
    model, add_s = _timed(
        lambda: EpistemicNearestNeighbors(
            train_x,
            np.asarray(train_y, dtype=float).reshape(-1, 1),
            scale_x=ENNScaleX.OFF,
            metric_learning=ENNMetricLearning.NONE,
            index_driver=ENNIndexDriver.FLAT,
        )
    )
    pair, query_s = _timed(lambda: _posterior_pair(model, _flat_test(test_x)))
    return pair[0], pair[1], add_s, query_s


def _auto_spans(n_rows: int) -> list[tuple[int, int]]:
    if n_rows <= 1:
        return [(0, n_rows)]
    step = min(AUTO_BATCH, n_rows // 2)
    stops = list(range(step, n_rows, step))
    stops.append(n_rows)
    start = 0
    spans: list[tuple[int, int]] = []
    for stop in stops:
        spans.append((start, stop))
        start = stop
    return spans


def _load_bpann_auto(train_x: np.ndarray, train_y: np.ndarray, work_dir: str):
    spans = _auto_spans(int(train_x.shape[0]))
    lo, hi = spans[0]
    model = EpistemicNearestNeighbors(
        train_x[lo:hi],
        train_y[lo:hi],
        scale_x=ENNScaleX.OFF,
        metric_learning=ENNMetricLearning.AUTO,
        index_driver=ENNIndexDriver.BPANN_DISK,
        work_dir=work_dir,
        enn_storage=ENNStorage.DISK,
    )
    for lo, hi in spans[1:]:
        model.add(train_x[lo:hi], train_y[lo:hi])
    model.ensure_index_sync()
    return model


def predict_bpann_auto(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray):
    y = np.asarray(train_y, dtype=float).reshape(-1, 1)
    with tempfile.TemporaryDirectory(prefix="qa_ts_bpann_") as work_dir:
        model, add_s = _timed(lambda: _load_bpann_auto(train_x, y, work_dir))
        pair, query_s = _timed(lambda: _posterior_pair(model, _flat_test(test_x)))
        return pair[0], pair[1], add_s, query_s


def _score_pair(
    name: str,
    y: np.ndarray,
    mu: np.ndarray,
    se: np.ndarray,
    elapsed: float,
    add_s: float,
    query_s: float,
):
    loglik, nrmse = loglik_and_nrmse(y, mu, se)
    return {
        f"loglik_{name}": loglik,
        f"nrmse_{name}": nrmse,
        f"seconds_{name}": float(elapsed),
        f"add_s_{name}": float(add_s),
        f"query_s_{name}": float(query_s),
    }


def run_one(kind: str, n_obs: int, seed: int, noise: float = NOISE) -> dict:
    data = assemble(kind, int(n_obs), int(seed), float(noise))
    if data["train_x"].shape[0] == 0 or data["test_y"].shape[0] == 0:
        raise RuntimeError(f"no train/test windows for {kind} n={n_obs}")
    y = data["test_y"]
    started = time.perf_counter()
    mu, se, add_s, query_s = predict_ennts(
        data["series"], data["test_x"], data["t_cut"]
    )
    ts_row = _score_pair(
        "ennts", y, mu, se, time.perf_counter() - started, add_s, query_s
    )
    started = time.perf_counter()
    learned = predict_ennts_metlrn(data["series"], data["test_x"], data["t_cut"])
    met_row = _score_pair(
        "ennts_metlrn",
        y,
        learned["mu"],
        learned["se"],
        time.perf_counter() - started,
        learned["add_s"],
        learned["query_s"],
    )
    started = time.perf_counter()
    mu, se, add_s, query_s = predict_enn(
        data["train_x"], data["train_y"], data["test_x"]
    )
    enn_row = _score_pair(
        "enn", y, mu, se, time.perf_counter() - started, add_s, query_s
    )
    started = time.perf_counter()
    mu, se, add_s, query_s = predict_bpann_auto(
        data["train_x"], data["train_y"], data["test_x"]
    )
    auto_row = _score_pair(
        "bpann_auto", y, mu, se, time.perf_counter() - started, add_s, query_s
    )
    counts = data["counts"]
    row = {
        "function": kind,
        "n": int(n_obs),
        "n_times": int(data["n_times"]),
        "n_zero": int(np.sum(counts == 0)),
        "n_train": int(data["train_x"].shape[0]),
        "n_test": int(y.shape[0]),
        "mean_count": float(np.mean(counts)),
    }
    row.update(ts_row)
    row.update(met_row)
    row.update(enn_row)
    row.update(auto_row)
    gain = learned["heldout_gain"]
    row["heldout_gain_metlrn"] = None if gain is None else float(gain)
    weights = learned["weights"]
    row["weights_metlrn"] = None if weights is None else weights.tolist()
    row["morph_remaining_metlrn"] = int(learned["morph_remaining"])
    row["uses_learned_metlrn"] = bool(learned["uses_learned"])
    return row


def _echo(row: dict) -> None:
    fields = (
        ("LARGER", "loglik_ennts"),
        ("SMALLER", "nrmse_ennts"),
        ("LARGER", "loglik_ennts_metlrn"),
        ("SMALLER", "nrmse_ennts_metlrn"),
        ("LARGER", "loglik_enn"),
        ("SMALLER", "nrmse_enn"),
        ("LARGER", "loglik_bpann_auto"),
        ("SMALLER", "nrmse_bpann_auto"),
        ("SMALLER", "add_s_ennts"),
        ("SMALLER", "query_s_ennts"),
        ("SMALLER", "add_s_ennts_metlrn"),
        ("SMALLER", "query_s_ennts_metlrn"),
        ("SMALLER", "add_s_enn"),
        ("SMALLER", "query_s_enn"),
        ("SMALLER", "add_s_bpann_auto"),
        ("SMALLER", "query_s_bpann_auto"),
    )
    metrics = " ".join(f"{tag}({name}) = {row[name]:.4f}" for tag, name in fields)
    print(
        "EVAL: "
        f"function = {row['function']} "
        f"n = {row['n']} "
        f"{metrics} "
        f"n_test = {row['n_test']}",
        flush=True,
    )


def run_sweep(kind: str, sizes: list[int], seed: int, out_dir: str) -> list[dict]:
    dest_dir = Path(out_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{kind}.json"
    rows: list[dict] = []
    for n_obs in sizes:
        row = run_one(kind, int(n_obs), int(seed), NOISE)
        rows.append(row)
        dest.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        _echo(row)
    return rows
