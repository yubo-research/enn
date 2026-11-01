from __future__ import annotations

import numpy as np
import pytest

from enn import (
    EpistemicNearestNeighbors,
    create_optimizer,
    turbo_zero_config,
)
from enn.enn.enn_params import ENNParams
from evals.harness.work_dirs import fresh_work_dir

PARAMS = ENNParams(3, 1.0, 0.1)


def _model() -> EpistemicNearestNeighbors:
    rng = np.random.default_rng(0)
    x = rng.uniform(size=(20, 3))
    return EpistemicNearestNeighbors(
        x, x.sum(axis=1, keepdims=True), work_dir=fresh_work_dir()
    )


@pytest.mark.parametrize(
    ("train_x", "train_y", "name", "shape"),
    [
        (np.zeros(3), np.zeros((1, 1)), "train_x", r"\[3\]"),
        (np.zeros((4, 3)), np.zeros(4), "train_y", r"\[4\]"),
    ],
)
def test_model_constructor_and_add_name_the_bad_argument(
    train_x, train_y, name, shape
) -> None:
    message = f"{name} must be 2-dimensional, got shape {shape}"
    with pytest.raises(ValueError, match=message):
        EpistemicNearestNeighbors(train_x, train_y, work_dir=fresh_work_dir())
    with pytest.raises(ValueError, match=message):
        _model().add(train_x, train_y)


def test_model_add_names_bad_train_yvar() -> None:
    with pytest.raises(ValueError, match="train_yvar must be 2-dimensional"):
        _model().add(np.zeros((2, 3)), np.zeros((2, 1)), np.zeros(2))


@pytest.mark.parametrize("call", ["posterior", "draw"])
def test_query_methods_reject_one_dimensional_x(call) -> None:
    model = _model()
    with pytest.raises(ValueError, match="x must be 2-dimensional"):
        if call == "posterior":
            model.posterior(np.zeros(3), params=PARAMS)
        else:
            model.posterior_function_draw(
                np.zeros(3), params=PARAMS, function_seeds=[1]
            )


def test_whatif_arguments_are_named() -> None:
    model = _model()
    q = np.zeros((1, 3))
    with pytest.raises(ValueError, match="y_whatif must be 2-dimensional"):
        model.conditional_posterior(q, np.zeros(1), q, params=PARAMS)
    with pytest.raises(ValueError, match="x_whatif must be 2-dimensional"):
        model.conditional_posterior(np.zeros(3), np.zeros((1, 1)), q, params=PARAMS)


def test_lists_convert_type_but_not_shape() -> None:
    model = _model()
    listed = model.posterior([[0.5, 0.5, 0.5]], params=PARAMS)
    arrayed = model.posterior(np.full((1, 3), 0.5), params=PARAMS)
    np.testing.assert_array_equal(listed.mu, arrayed.mu)
    with pytest.raises(ValueError, match="x must be 2-dimensional"):
        model.posterior([0.5, 0.5, 0.5], params=PARAMS)


def test_optimizer_tell_rejects_one_dimensional_y_and_returns_matrix() -> None:
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    opt = create_optimizer(
        bounds=bounds,
        config=turbo_zero_config(num_init=2),
        seed=0,
        work_dir=fresh_work_dir(),
    )
    x = opt.ask(2)
    with pytest.raises(ValueError, match="train_y must be 2-dimensional"):
        opt.tell(x, np.zeros(2))
    y = np.zeros((2, 1))
    told = opt.tell(x, y)
    assert told.shape == (2, 1)


def test_subsample_loglik_rejects_one_dimensional_y() -> None:
    from enn.enn.enn_fitter import subsample_loglik

    with pytest.raises(ValueError, match="y must be 2-dimensional"):
        subsample_loglik(
            _model(),
            np.zeros((4, 3)),
            np.zeros(4),
            paramss=[PARAMS],
            seed=0,
        )


def test_params_are_keyword_only() -> None:
    from enn.enn.affine_calibrator import fit_affine_calibrator

    model = _model()
    q = np.zeros((1, 3))
    with pytest.raises(TypeError):
        model.batch_posterior(q, [PARAMS])
    with pytest.raises(TypeError):
        model.posterior_function_draw(q, PARAMS, function_seeds=[1])
    with pytest.raises(TypeError):
        fit_affine_calibrator(model, PARAMS, num_samples=4, seed=0)


def test_observation_arguments_use_train_names() -> None:
    model = _model()
    x, y = np.zeros((1, 3)), np.zeros((1, 1))
    model.add(train_x=x, train_y=y)
    assert len(model) == 21
    opt = create_optimizer(
        bounds=np.array([[0.0, 1.0]]),
        config=turbo_zero_config(num_init=1),
        seed=0,
        work_dir=fresh_work_dir(),
    )
    arm = opt.ask(1)
    opt.tell(train_x=arm, train_y=y)
    with pytest.raises(ValueError, match="train_yvar must be omitted on every tell"):
        opt.tell(train_x=arm, train_y=y, train_yvar=np.ones((1, 1)))
