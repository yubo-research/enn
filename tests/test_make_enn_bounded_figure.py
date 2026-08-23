from __future__ import annotations

from unittest import mock

import numpy as np

from make_enn_bounded_figure import (
    compute_figure_ylim,
    fit_model,
    LO,
    HI,
    main,
    plot_panel,
    Y_PAD_MIN,
)
from ops.qa import make_bounded_1d_xy, y_bounds_array


def test_fit_model_returns_model_and_params():
    rng = np.random.default_rng(0)
    x_train, y_train = make_bounded_1d_xy(8, rng, -3.0, 7.0)
    model, params = fit_model(x_train, y_train, y_bounds=None)
    assert model is not None
    assert params is not None
    assert len(model) == len(x_train)


def test_compute_figure_ylim_extends_beyond_bounds_and_covers_spill():
    rng = np.random.default_rng(1)
    x_train, y_train = make_bounded_1d_xy(8, rng, -3.0, 7.0)
    x_grid = np.linspace(0.0, 1.0, 50).reshape(-1, 1)
    ylo, yhi = compute_figure_ylim(x_train, y_train, x_grid)
    assert ylo < LO - Y_PAD_MIN
    assert yhi > HI + Y_PAD_MIN
    model, params = fit_model(x_train, y_train, None)
    post = model.posterior(x_grid, params=params)
    lower, upper = post.confidence_interval(0.95)
    assert ylo <= lower[:, 0].min()
    assert yhi >= upper[:, 0].max()


def test_plot_panel_draws_bounded_posterior():
    rng = np.random.default_rng(1)
    x_train, y_train = make_bounded_1d_xy(8, rng, -3.0, 7.0)
    x_grid = np.linspace(0.0, 1.0, 20).reshape(-1, 1)
    ylim = compute_figure_ylim(x_train, y_train, x_grid)
    ax = mock.MagicMock()
    plot_panel(
        ax,
        x_train=x_train,
        y_train=y_train,
        x_grid=x_grid,
        y_bounds=y_bounds_array(-3.0, 7.0),
        title="Bounded",
        ylim=ylim,
    )
    ax.fill_between.assert_called_once()
    ax.plot.assert_called()
    ax.scatter.assert_called_once()
    assert ax.axhline.call_count == 2
    for call in ax.axhline.call_args_list:
        assert call.kwargs["linestyle"] == "--"
    ax.set_ylim.assert_called_once_with(ylim)


def test_main_writes_figure(tmp_path, monkeypatch):
    out = tmp_path / "enn_figure.pdf"
    monkeypatch.setattr("make_enn_bounded_figure.OUTPUT_PDF", str(out))
    with mock.patch("matplotlib.pyplot.subplots") as subplots:
        fig = mock.MagicMock()
        axes = (mock.MagicMock(), mock.MagicMock())
        subplots.return_value = (fig, axes)
        main()
    fig.savefig.assert_called_once_with(str(out), bbox_inches="tight")
