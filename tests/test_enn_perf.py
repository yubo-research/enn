from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from enn import ENNStatefulFitter, EpistemicNearestNeighbors
from enn.enn.enn_class_support import (
    enn_index_neighbor_distances_and_indices,
    enn_neighbor_distances_and_indices,
)
from enn.enn.enn_params import ENNParams, PosteriorFlags

_POSTERIOR_SPEED_BASELINE = (
    Path(__file__).resolve().parent / "fixtures" / "posterior_speed_baseline.json"
)


def make_enn_demo_data(num_samples: int, k: int, noise: float, m: int = 1):
    x = np.sort(np.random.rand(num_samples + 4))
    x[-3] = x[-4]
    x[-2] = x[-4]
    x[-1] = x[-4]
    x[1] = x[0] + 0.03
    eps = np.random.randn(num_samples + 4)
    y = np.sin(2 * m * np.pi * x) + noise * eps
    yvar = (noise**2) * np.ones_like(y)
    model = EpistemicNearestNeighbors(x[:, None], y[:, None], yvar[:, None])
    rng = np.random.default_rng(0)
    fitter = ENNStatefulFitter(k=k, rng=rng)
    fitter.tell(model.train_x, model.train_y, model.train_yvar)
    params = fitter.ask(
        model,
        num_fit_candidates=100,
        num_fit_samples=min(10, num_samples),
    )
    return x, y, model, params


def plot_enn_posterior_logic(model, params):
    x_hat = np.linspace(0.0, 1.0, 30)
    x_hat_2d = x_hat[:, None]
    posterior = model.posterior(
        x_hat_2d, params=params, flags=PosteriorFlags(exclude_nearest=False)
    )
    return posterior.mu[:, 0], posterior.se[:, 0]


@pytest.mark.skip(reason="Timing unreliable across machines; run manually to verify")
def test_enn_demo_performance():
    np.random.seed(1)
    t0 = time.time()
    x, y, model, params = make_enn_demo_data(num_samples=1_000_000, k=5, noise=0.3, m=3)
    mu, se = plot_enn_posterior_logic(model, params)
    elapsed = time.time() - t0
    print(f"\nTime taken: {elapsed:.3f} seconds")
    assert elapsed < 0.3, f"Expected < 0.3s, got {elapsed:.3f}s"


def _median_seconds(fn, *, reps: int = 12) -> float:
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


def _neighbor_lookup_ratio(n: int, *, d: int = 1, k: int = 10, seed: int = 42) -> float:
    """Wall-clock ratio: index_search (f64 exact) vs FAISS-only neighbor lookup."""
    rng = np.random.default_rng(seed)
    train_x = rng.standard_normal((n, d))
    train_y = rng.normal(0.0, 100.0, size=(n, d))
    model = EpistemicNearestNeighbors(train_x, train_y, train_yvar=None)
    rust = model.rust_backend

    def index_search_neighbors() -> None:
        enn_index_neighbor_distances_and_indices(
            rust,
            train_x,
            search_k=k,
            exclude_nearest=False,
        )

    def faiss_batch_neighbors() -> None:
        enn_neighbor_distances_and_indices(
            rust,
            train_x,
            search_k=k,
            exclude_nearest=False,
        )

    for _ in range(3):
        index_search_neighbors()
        faiss_batch_neighbors()

    t_index = _median_seconds(index_search_neighbors)
    t_faiss = _median_seconds(faiss_batch_neighbors)
    return t_index / max(t_faiss, 1e-12)


@pytest.mark.parametrize(
    ("n", "max_ratio"),
    [
        (20, 2.5),  # user-reported scenario (n=20, k=10)
        (500, 3.0),  # n_query=n_train; batched f64 matrix should track FAISS
    ],
)
def test_index_search_neighbor_lookup_not_much_slower_than_faiss_only(
    n: int, max_ratio: float
):
    """index_search (f64 exact) must not regress far beyond FAISS-only lookup."""
    ratio = _neighbor_lookup_ratio(n)
    assert ratio <= max_ratio, (
        f"index_search must be <= {max_ratio}x FAISS-only at n={n}, "
        f"got {ratio:.2f}x (slowdown bad)"
    )


def _posterior_self_search_median_seconds(
    *, tie_break_neighbors: bool, reps: int = 8
) -> float:
    rng = np.random.default_rng(0)
    n, d, m = 1024, 10, 5
    x = rng.standard_normal((n, d))
    y = rng.normal(0.0, 100.0, size=(n, m))
    model = EpistemicNearestNeighbors(x, y)
    params = ENNParams(
        k_num_neighbors=10, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.1
    )
    flags = PosteriorFlags(tie_break_neighbors=tie_break_neighbors)

    def run() -> None:
        model.posterior(x, params=params, flags=flags)

    for _ in range(3):
        run()
    return _median_seconds(run, reps=reps)


def test_posterior_self_search_tie_break_not_much_slower_than_no_tie_break():
    """posterior(train_x) tie-break-on must track tie-break-off at n=1024."""
    t_off = _posterior_self_search_median_seconds(tie_break_neighbors=False)
    t_on = _posterior_self_search_median_seconds(tie_break_neighbors=True)
    ratio = t_on / max(t_off, 1e-12)
    assert ratio <= 1.15, (
        f"tie-break posterior must be <= 1.15x no-tie-break at n=1024, "
        f"got {ratio:.2f}x (t_on={t_on:.4f}s t_off={t_off:.4f}s)"
    )


def _lattice_posterior_self_search_median_seconds(
    *, tie_break_neighbors: bool, reps: int = 8
) -> float:
    """1D grid tiled to d=10; tie-heavy self-search (regression from PR #29)."""
    n, d = 1024, 10
    x = np.linspace(0.0, 1.0, n)[:, None]
    x = np.tile(x, (1, d))
    y = np.sin(8.0 * np.pi * x[:, 0:1])
    model = EpistemicNearestNeighbors(x, y)
    params = ENNParams(
        k_num_neighbors=10, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.1
    )
    flags = PosteriorFlags(tie_break_neighbors=tie_break_neighbors)

    def run() -> None:
        model.posterior(x, params=params, flags=flags)

    for _ in range(3):
        run()
    return _median_seconds(run, reps=reps)


def test_lattice_posterior_self_search_tie_break_not_much_slower_than_no_tie_break():
    """lattice posterior(x_train) tie-break-on must track tie-break-off at n=1024."""
    t_off = _lattice_posterior_self_search_median_seconds(tie_break_neighbors=False)
    t_on = _lattice_posterior_self_search_median_seconds(tie_break_neighbors=True)
    ratio = t_on / max(t_off, 1e-12)
    assert ratio <= 1.15, (
        f"lattice tie-break posterior must be <= 1.15x no-tie-break at n=1024, "
        f"got {ratio:.2f}x (t_on={t_on:.4f}s t_off={t_off:.4f}s)"
    )


def test_index_search_slowdown_does_not_blow_up_with_n():
    """Ratio should stay bounded as n_query=n_train grows."""
    n_small, n_large = 512, 4096
    ratio_small = _neighbor_lookup_ratio(n_small, seed=7)
    ratio_large = _neighbor_lookup_ratio(n_large, seed=7)
    assert ratio_large <= max(ratio_small * 1.5, 3.0), (
        f"expected bounded slowdown ratio: n={n_small} -> {ratio_small:.2f}x, "
        f"n={n_large} -> {ratio_large:.2f}x"
    )


def _self_search_timing_ratios(
    scenario: dict,
    *,
    tie_break_neighbors: bool,
    reps: int = 10,
) -> tuple[float, float, float, float, float]:
    """Return (t_post, t_index, t_faiss, index/faiss, posterior/index) medians."""
    rng = np.random.default_rng(int(scenario["seed"]))
    n, d, m, k = scenario["n"], scenario["d"], scenario["m"], scenario["k"]
    x = rng.standard_normal((n, d))
    y = rng.normal(0.0, 100.0, size=(n, m))
    model = EpistemicNearestNeighbors(x, y)
    params = ENNParams(
        k_num_neighbors=k, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.1
    )
    flags = PosteriorFlags(tie_break_neighbors=tie_break_neighbors)
    rust = model.rust_backend

    def posterior() -> None:
        model.posterior(x, params=params, flags=flags)

    def index_neighbors() -> None:
        enn_index_neighbor_distances_and_indices(
            rust,
            x,
            search_k=k,
            exclude_nearest=False,
            tie_break_neighbors=tie_break_neighbors,
        )

    def faiss_neighbors() -> None:
        enn_neighbor_distances_and_indices(
            rust, x, search_k=k, exclude_nearest=False
        )

    for _ in range(3):
        posterior()
        index_neighbors()
        faiss_neighbors()

    t_post = _median_seconds(posterior, reps=reps)
    t_index = _median_seconds(index_neighbors, reps=reps)
    t_faiss = _median_seconds(faiss_neighbors, reps=reps)
    index_vs_faiss = t_index / max(t_faiss, 1e-12)
    posterior_vs_index = t_post / max(t_index, 1e-12)
    return t_post, t_index, t_faiss, index_vs_faiss, posterior_vs_index


@pytest.mark.parametrize("tie_break_neighbors", [False, True])
def test_posterior_self_search_not_slower_than_9eaa27_baseline(tie_break_neighbors: bool):
    """Self-search: index_search vs FAISS and posterior stats overhead stay bounded."""
    with open(_POSTERIOR_SPEED_BASELINE) as f:
        baseline = json.load(f)
    for scenario in baseline["scenarios"]:
        t_post, t_index, t_faiss, index_vs_faiss, posterior_vs_index = (
            _self_search_timing_ratios(
                scenario, tie_break_neighbors=tie_break_neighbors
            )
        )
        max_index_vs_faiss = float(scenario["max_index_vs_faiss"])
        max_posterior_vs_index = float(scenario["max_posterior_vs_index"])
        assert index_vs_faiss <= max_index_vs_faiss, (
            f"{scenario['name']} tie_break={tie_break_neighbors}: "
            f"index/faiss={index_vs_faiss:.3f}x exceeds cap {max_index_vs_faiss:.2f}x "
            f"(t_index={t_index:.4f}s t_faiss={t_faiss:.4f}s; "
            f"baseline_rev={baseline['baseline_revision'][:7]})"
        )
        assert posterior_vs_index <= max_posterior_vs_index, (
            f"{scenario['name']} tie_break={tie_break_neighbors}: "
            f"posterior/index={posterior_vs_index:.3f}x exceeds cap "
            f"{max_posterior_vs_index:.2f}x "
            f"(t_post={t_post:.4f}s t_index={t_index:.4f}s; "
            f"baseline_rev={baseline['baseline_revision'][:7]})"
        )
