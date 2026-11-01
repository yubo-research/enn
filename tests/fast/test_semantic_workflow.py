"""One workflow: add, enn_fit with the token, posterior, draws."""

from dataclasses import replace

import numpy as np
import pytest

from enn import (
    ENNOutputBlend,
    ENNStatefulFitter,
    EpistemicNearestNeighbors,
    SemanticInputTransform,
    SemanticOutputProjection,
)
from enn.enn.enn_fit import enn_fit, subsample_loglik
from enn.enn.enn_params import ENNParams, PosteriorFlags
from evals.harness.work_dirs import fresh_work_dir

NAMES = ("a", "b", "c")
BOUNDS = np.array([[0.0, 1.0]] * 3)
SHAPES = ["none", "xx", "xy", "yy", "xx+yy"]


class Encoder:
    def encode(self, texts):
        rows = {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [1.0, 1.0], "y": [1.0, 0.0]}
        return np.array([rows[text.partition("::")[0]] for text in texts])


def input_transform(kind):
    if kind == "xy":
        return SemanticInputTransform.from_input_output(
            NAMES, BOUNDS, ["y"], encoder=Encoder()
        )
    return SemanticInputTransform(NAMES, BOUNDS, encoder=Encoder())


def output_projection(outputs=3):
    return SemanticOutputProjection.from_embeddings(
        [f"y{i}" for i in range(outputs)],
        [[1.0, 0.0]] * (outputs - 1) + [[0.0, 1.0]],
    )


def data(n=60, outputs=3, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(size=(n, 3))
    signal = np.sin(3 * x[:, 0])
    columns = [signal + 0.01 * (j + 1) * rng.normal(size=n) for j in range(outputs)]
    return x, np.column_stack(columns)


def build(x, y, shape="none", work_dir=None, **kwargs):
    if "xx" in shape or "xy" in shape:
        kwargs["input_transform"] = input_transform("xy" if "xy" in shape else "xx")
    if "yy" in shape:
        kwargs["output_projection"] = output_projection(y.shape[1])
    return EpistemicNearestNeighbors(
        x, y, work_dir=fresh_work_dir() if work_dir is None else work_dir, **kwargs
    )


def fit(model, token, seed=0, **kwargs):
    return enn_fit(
        model,
        k=4,
        num_fit_candidates=5,
        rng=np.random.default_rng(seed),
        incremental=token,
        **kwargs,
    )


@pytest.mark.parametrize("shape", SHAPES)
def test_workflow_over_batches(shape, tmp_path):
    x, y = data()
    model = build(x[:0], y[:0], shape, tmp_path)
    rng = np.random.default_rng(1)
    for start in range(0, 60, 20):
        batch = slice(start, start + 20)
        params = enn_fit(
            model,
            k=4,
            num_fit_candidates=5,
            rng=rng,
            incremental=model.add(x[batch], y[batch]),
        )
        assert params.k_num_neighbors == 4
        post = model.posterior(x[:5], params=params)
        draws, _ = model.posterior_function_draw(x[:5], params, function_seeds=[1, 2])
        assert np.isfinite(post.mu).all() and (post.se > 0).all()
        assert draws.shape == (5, 3, 2) and np.isfinite(draws).all()
        assert (params.output_blend is not None) == ("yy" in shape)


@pytest.mark.parametrize(
    "shape, mean, se",
    [
        ("none", 0.9165882494609465, 0.3959602534604401),
        ("xx", 0.9256748294424608, 0.4048810084965212),
        ("xy", 0.9245619139239245, 0.4033339201238322),
    ],
)
def test_models_without_yy_match_the_values_from_before_the_blend(shape, mean, se):
    x, y = data()
    model = build(x[:0], y[:0], shape)
    params = fit(model, model.add(x, y))
    assert params.output_blend is None
    assert params.epistemic_variance_scale == pytest.approx(448.174093090275, rel=1e-12)
    assert params.aleatoric_variance_scale == pytest.approx(
        2.3232730047134242, rel=1e-12
    )
    post = model.posterior(x[:1], params=params)
    assert post.mu[0, 0] == pytest.approx(mean, rel=1e-12)
    assert post.se[0, 2] == pytest.approx(se, rel=1e-12)


def test_enn_fit_and_a_held_fitter_both_return_a_blend():
    x, y = data()
    model = build(x[:0], y[:0], "yy")
    token_params = fit(model, model.add(x, y))
    fitter = ENNStatefulFitter(k=4, seed=0)
    fitter.tell(x, y)
    held_params = fitter.ask(model, num_fit_candidates=5)
    assert token_params.output_blend is not None
    assert held_params.output_blend is not None


@pytest.mark.parametrize("shape", SHAPES)
def test_ask_leaves_the_model_unchanged(shape):
    x, y = data()
    model = build(x, y, shape)
    probe = ENNParams(4, 1, 0)
    before = model.posterior(x[:5], params=probe)
    scale = model._x_scale.copy()
    transform = model.input_transform
    fitter = ENNStatefulFitter(k=4, seed=0, input_transform=transform)
    fitter.tell(x, y)
    fitter.ask(model, num_fit_candidates=5)
    assert len(model) == 60
    np.testing.assert_array_equal(scale, model._x_scale)
    after = model.posterior(x[:5], params=probe)
    np.testing.assert_array_equal(before.mu, after.mu)
    np.testing.assert_array_equal(before.se, after.se)


def test_unblended_params_predict_unblended_on_a_yy_model():
    x, y = data()
    plain = build(x, y)
    configured = build(x, y, "yy")
    params = ENNParams(4, 1, 0.1)
    np.testing.assert_array_equal(
        plain.posterior(x[:5], params=params).mu,
        configured.posterior(x[:5], params=params).mu,
    )


def test_batch_posterior_applies_each_entrys_blend():
    x, y = data()
    model = build(x[:0], y[:0], "yy", output_alpha=0.1)
    low = fit(model, model.add(x, y))
    high_blend = ENNOutputBlend(
        np.full(3, 0.9),
        low.output_blend.projector,
        low.output_blend.center,
        low.output_blend.scale,
    )
    high = replace(low, output_blend=high_blend)
    batch = model.batch_posterior(x[:5], [low, high])
    for entry, params in enumerate((low, high)):
        one = model.posterior(x[:5], params=params)
        np.testing.assert_allclose(batch.mu[entry], one.mu, atol=1e-12)
        np.testing.assert_allclose(batch.se[entry], one.se, atol=1e-12)
    assert not np.allclose(batch.mu[0], batch.mu[1])
    low_score, high_score = subsample_loglik(
        model, x, y, paramss=[low, high], P=20, rng=np.random.default_rng(0)
    )
    assert low_score != high_score


def test_sign_opposed_output_gets_low_alpha_and_others_do_not():
    rng = np.random.default_rng(30)
    x = rng.uniform(size=(300, 2))
    signal = np.sin(3 * x[:, 0])
    y = np.column_stack((signal, signal, -signal)) + rng.normal(0, 0.1, (300, 3))
    projection = SemanticOutputProjection.from_embeddings(["a", "b", "c"], [[1, 0]] * 3)
    model = EpistemicNearestNeighbors(
        x[:0], y[:0], output_projection=projection, work_dir=fresh_work_dir()
    )
    alpha = fit(model, model.add(x, y)).output_blend.alpha
    assert alpha[2] < 0.2 and alpha[0] > alpha[2] and alpha[1] > alpha[2]


def test_constant_output_is_unchanged_and_does_not_affect_others():
    x, y = data()
    constant = y.copy()
    constant[:, 2] = 0.1
    model = build(x[:0], constant[:0], "yy", output_alpha=1.0)
    params = fit(model, model.add(x, constant))
    blend = params.output_blend
    assert blend.alpha[2] == 0 and blend.scale[2] == 1
    post = model.posterior(x[:5], params=params)
    np.testing.assert_allclose(post.mu[:, 2], 0.1, atol=1e-12)
    other = model.posterior(x[:5], params=replace(params, output_blend=None))
    z = (other.mu - blend.center) / blend.scale
    expected = z[:, :2] @ blend.projector[:2, :2] * blend.scale[:2]
    np.testing.assert_allclose(post.mu[:, :2], expected + blend.center[:2], atol=1e-9)


def test_pinned_alpha_stays_across_asks():
    x, y = data()
    model = build(x[:0], y[:0], "yy", output_alpha=0.1)
    rng = np.random.default_rng(0)
    for start in (0, 20, 40):
        token = model.add(x[start : start + 20], y[start : start + 20])
        params = enn_fit(model, k=4, num_fit_candidates=3, rng=rng, incremental=token)
        np.testing.assert_array_equal(params.output_blend.alpha, np.full(3, 0.1))


def test_output_blend_round_trips_and_validates():
    x, y = data()
    model = build(x[:0], y[:0], "yy")
    blend = fit(model, model.add(x, y)).output_blend
    copy = ENNOutputBlend(blend.alpha, blend.projector, blend.center, blend.scale)
    assert copy == blend
    params = ENNParams(4, 1, 0, output_blend=blend)
    assert params == ENNParams(4, 1, 0, output_blend=copy)
    assert hash(params) == hash(ENNParams(4, 1, 0))
    assert params != ENNParams(4, 1, 0)
    with pytest.raises(ValueError, match="scale"):
        ENNOutputBlend(blend.alpha, blend.projector, blend.center, blend.scale * 0)


def test_warm_blend_with_wrong_outputs_is_an_error():
    x, y = data()
    model = build(x[:0], y[:0], "yy")
    token = model.add(x, y)
    wrong = ENNOutputBlend(np.zeros(1), np.eye(1), np.zeros(1), np.ones(1))
    with pytest.raises(ValueError, match="warm output blend"):
        fit(model, token, params_warm_start=ENNParams(4, 1, 0, output_blend=wrong))


def test_enn_fit_requires_the_token():
    x, y = data()
    model = build(x, y)
    with pytest.raises(TypeError, match="incremental"):
        enn_fit(model, k=4, num_fit_candidates=3, rng=np.random.default_rng(0))
    with pytest.raises(TypeError, match="token"):
        fit(model, None)


def test_one_rng_works_across_calls_and_is_reproducible():
    x, y = data()

    def run():
        model = build(x[:0], y[:0])
        rng = np.random.default_rng(7)
        results = []
        for start in (0, 20, 40):
            token = model.add(x[start : start + 20], y[start : start + 20])
            results.append(
                enn_fit(model, k=4, num_fit_candidates=5, rng=rng, incremental=token)
            )
        return results

    assert run() == run()


@pytest.mark.parametrize("bounded", [False, True])
def test_constructor_rows_count_like_told_rows(bounded):
    x, y = data(outputs=1)
    y = (y - y.min()) / (y.max() - y.min()) * 0.8 + 0.1
    bounds = np.array([[0.0, 1.0]]) if bounded else None
    model = build(x[:40], y[:40], y_bounds=bounds)
    token = model.add(x[40:], y[40:])
    token_params = enn_fit(
        model,
        k=4,
        num_fit_candidates=5,
        rng=np.random.default_rng(3),
        incremental=token,
    )
    fitter = ENNStatefulFitter(k=4, rng=np.random.default_rng(3))
    fitter.tell(x, y, y_bounds=bounds)
    assert token_params == fitter.ask(model, num_fit_candidates=5)


def test_disk_reopen_predicts_the_same_and_warm_blend_survives(tmp_path):
    x, y = data()
    kwargs = {
        "work_dir": tmp_path,
        "output_projection": output_projection(),
    }
    model = EpistemicNearestNeighbors(x[:0], y[:0], **kwargs)
    params = fit(model, model.add(x[:8], y[:8]))
    expected = model.posterior(x[:5], params=params)
    model.persist_index_to_disk()
    del model
    reopened = EpistemicNearestNeighbors(np.empty((0, 3)), np.empty((0, 3)), **kwargs)
    actual = reopened.posterior(x[:5], params=params)
    np.testing.assert_allclose(actual.mu, expected.mu, atol=1e-12)
    np.testing.assert_allclose(actual.se, expected.se, atol=1e-12)
    token = reopened.add(x[8:9], y[8:9])
    refit = fit(reopened, token, params_warm_start=params)
    assert refit.output_blend == params.output_blend


def test_observation_noise_flag_uses_the_blend():
    x, y = data()
    model = build(x[:0], y[:0], "yy", output_alpha=0.9)
    params = fit(model, model.add(x, y))
    flags = PosteriorFlags(observation_noise=True)
    blended = model.posterior(x[:5], params=params, flags=flags)
    plain = model.posterior(
        x[:5], params=replace(params, output_blend=None), flags=flags
    )
    assert not np.allclose(blended.mu, plain.mu)
