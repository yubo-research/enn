#!/usr/bin/env python3
"""Generate side-by-side ENN posterior figure with noise-field samples."""

from __future__ import annotations

import numpy as np

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_fit import enn_fit
from enn.enn.posterior_flags import PosteriorFlags
from ops.qa import make_bounded_1d_xy, y_bounds_array

LO, HI = -3.0, 7.0
N_OBS = 12
SEED = 42
Y_SCALE = 2.0
Y_CENTER = -1.0
K = 5
NUM_FUNCTION_DRAWS = 40
X_GRID_SIZE = 200
OUTPUT_PDF = "enn_figure.pdf"


def fit_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    y_bounds: np.ndarray | None,
) -> tuple[EpistemicNearestNeighbors, object]:
    model = EpistemicNearestNeighbors(x_train, y_train, y_bounds=y_bounds)
    params = enn_fit(
        model,
        k=K,
        num_fit_candidates=30,
        num_fit_samples=20,
        rng=np.random.default_rng(SEED + 1),
    )
    return model, params


def plot_panel(
    ax,
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_grid: np.ndarray,
    y_bounds: np.ndarray | None,
    title: str,
) -> None:
    model, params = fit_model(x_train, y_train, y_bounds)
    post = model.posterior(x_grid, params=params)
    mu = post.mu[:, 0]
    x_line = x_grid[:, 0]
    lower, upper = post.confidence_interval(0.95)
    band_lo = lower[:, 0]
    band_hi = upper[:, 0]

    seeds = list(range(NUM_FUNCTION_DRAWS))
    draws, _idx = model.posterior_function_draw(
        x_grid,
        params,
        function_seeds=seeds,
        flags=PosteriorFlags(),
    )

    ax.axhline(LO, color="0.5", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.axhline(HI, color="0.5", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.fill_between(x_line, band_lo, band_hi, color="tab:blue", alpha=0.2)
    for i in range(draws.shape[2]):
        ax.plot(
            x_line,
            draws[:, 0, i],
            color="black",
            alpha=0.35,
            linewidth=0.7,
            linestyle="--",
        )
    ax.plot(x_line, mu, linestyle="--", color="tab:blue", linewidth=1.2, alpha=0.9)
    ax.scatter(
        x_train[:, 0],
        y_train[:, 0],
        s=40,
        color="black",
        alpha=0.75,
        zorder=5,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(LO - 0.5, HI + 0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)


def main() -> None:
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(SEED)
    x_train, y_train = make_bounded_1d_xy(
        N_OBS, rng, LO, HI, y_scale=Y_SCALE, y_center=Y_CENTER
    )
    x_grid = np.linspace(0.0, 1.0, X_GRID_SIZE).reshape(-1, 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    plot_panel(
        axes[0],
        x_train=x_train,
        y_train=y_train,
        x_grid=x_grid,
        y_bounds=None,
        title="Unbounded ENN",
    )
    plot_panel(
        axes[1],
        x_train=x_train,
        y_train=y_train,
        x_grid=x_grid,
        y_bounds=y_bounds_array(LO, HI),
        title=r"Bounded ENN ($y\in(-3,7)$)",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    print(f"wrote {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
