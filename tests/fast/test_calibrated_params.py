from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from enn import (
    AcqType,
    AffineCalibrator,
    CandidateGenConfig,
    ENNFitConfig,
    ENNParams,
    ENNStatefulFitter,
    ENNSurrogateConfig,
    EpistemicNearestNeighbors,
    PosteriorFlags,
    create_optimizer,
    fit_affine_calibrator,
    turbo_enn_config,
)

GOLDEN = json.loads(
    (Path(__file__).parent / "golden" / "calibrated_params.json").read_text()
)
Q = np.array([[0.1, 0.2], [0.5, 0.5], [0.9, 0.3]])
FLAGS = {
    "default": PosteriorFlags(),
    "noise": PosteriorFlags(observation_noise=True),
    "excl": PosteriorFlags(exclude_nearest=True),
}
CASES = {
    "plain": {"m": 1, "bounded": False},
    "bounded": {"m": 1, "bounded": True},
    "two_metric": {"m": 2, "bounded": False},
}
ROUNDING = 1e-14


def _data(seed: int, m: int, bounded: bool) -> tuple[np.ndarray, np.ndarray]:
    r = np.random.default_rng(seed)
    x = r.uniform(size=(60, 2))
    y = 1.0 + 2.0 * x[:, :1] + 0.1 * r.standard_normal((60, 1))
    if m == 2:
        y = np.hstack([y, -y + 0.05 * r.standard_normal((60, 1))])
    if bounded:
        y = 0.02 + 0.96 * (y - y.min(0)) / (y.max(0) - y.min(0))
    return x, y


def _fitted(tmp_path, name: str):
    case = CASES[name]
    x, y = _data(3, case["m"], case["bounded"])
    kw = {"y_bounds": np.array([[0.0, 1.0]])} if case["bounded"] else {}
    model = EpistemicNearestNeighbors(x, y, work_dir=tmp_path / "store", **kw)
    params = ENNStatefulFitter(4, seed=7).ask(
        model, num_fit_candidates=8, num_fit_samples=20, affine_calibrate=True
    )
    return model, params


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("flag", list(FLAGS))
def test_posterior_with_calibrated_params_matches_captured_fitter_values(
    tmp_path, name, flag
) -> None:
    model, params = _fitted(tmp_path, name)
    golden = GOLDEN[name]
    np.testing.assert_array_equal(params.calibrator.a, golden["a"])
    np.testing.assert_array_equal(params.calibrator.b, golden["b"])
    np.testing.assert_array_equal(params.calibrator.c, golden["c"])
    post = model.posterior(Q, params=params, flags=FLAGS[flag])
    want = golden[f"posterior_{flag}"]
    np.testing.assert_array_equal(post.se_epi, want["se_epi"])
    np.testing.assert_array_equal(post.se_ale, want["se_ale"])
    np.testing.assert_allclose(post.se, want["se"], rtol=ROUNDING, atol=0)
    np.testing.assert_allclose(post.mu, want["mu"], rtol=ROUNDING, atol=0)


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("flag", list(FLAGS))
def test_draws_with_calibrated_params_match_captured_fitter_values(
    tmp_path, name, flag
) -> None:
    model, params = _fitted(tmp_path, name)
    draws, _ = model.posterior_function_draw(
        Q, params=params, function_seeds=[11, 12], flags=FLAGS[flag]
    )
    np.testing.assert_array_equal(draws, GOLDEN[name][f"draw_{flag}"])


def _trace(tmp_path, acq: AcqType, seed: int, bounded: bool) -> dict:
    r = np.random.default_rng(100 + seed)
    kw = {"y_bounds": np.array([[-10.0, 10.0]])} if bounded else {}
    fit = ENNFitConfig(num_fit_samples=20, num_fit_candidates=8, affine_calibrate=True)
    enn = ENNSurrogateConfig(k=4, fit=fit, work_dir=tmp_path / "store", **kw)
    config = turbo_enn_config(
        enn=enn,
        num_init=6,
        acq_type=acq,
        candidates=CandidateGenConfig(max_candidates=64),
    )
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    optimizer = create_optimizer(bounds=bounds, config=config, seed=seed)
    arms, told = [], []
    for _ in range(14):
        x = optimizer.ask(2)
        noise = 0.01 * r.standard_normal((2, 1))
        y = -np.sum((x - 0.4) ** 2, axis=1, keepdims=True) + noise
        told.append(np.asarray(optimizer.tell(x, y)).tolist())
        arms.append(x.tolist())
    return {"arms": arms, "told": told}


@pytest.mark.parametrize("acq", [AcqType.PARETO, AcqType.UCB, AcqType.THOMPSON])
@pytest.mark.parametrize("seed", [1, 2])
@pytest.mark.parametrize("bounded", [False, True])
def test_optimizer_affine_calibrate_results_are_unchanged(
    tmp_path, acq, seed, bounded
) -> None:
    got = _trace(tmp_path, acq, seed, bounded)
    assert got == GOLDEN["optimizer"][f"{acq.name}_{seed}_{bounded}"]


def _simple_model(tmp_path):
    rng = np.random.default_rng(1)
    x = rng.uniform(size=(30, 2))
    y = 1.0 + 2.0 * x[:, :1] + 0.1 * rng.standard_normal((30, 1))
    return EpistemicNearestNeighbors(x, y, work_dir=tmp_path / "store")


CAL = AffineCalibrator(a=np.array([0.5]), b=np.array([2.0]), c=np.array([3.0]))
BARE = ENNParams(3, 1.0, 0.1)
CALIBRATED = replace(BARE, calibrator=CAL)


def test_batch_posterior_applies_each_params_calibrator(tmp_path) -> None:
    model = _simple_model(tmp_path)
    batch = model.batch_posterior(Q, paramss=[BARE, CALIBRATED])
    bare = model.posterior(Q, params=BARE)
    calibrated = model.posterior(Q, params=CALIBRATED)
    np.testing.assert_array_equal(batch.mu[0], bare.mu)
    np.testing.assert_array_equal(batch.mu[1], calibrated.mu)
    np.testing.assert_array_equal(batch.se[1], calibrated.se)
    np.testing.assert_array_equal(batch.se_epi[1], calibrated.se_epi)
    np.testing.assert_array_equal(batch.se_ale[1], calibrated.se_ale)


def test_conditional_posterior_and_draw_apply_the_calibrator(tmp_path) -> None:
    model = _simple_model(tmp_path)
    x_whatif, y_whatif = Q[:1], np.array([[2.0]])
    bare = model.conditional_posterior(x_whatif, y_whatif, Q, params=BARE)
    calibrated = model.conditional_posterior(x_whatif, y_whatif, Q, params=CALIBRATED)
    np.testing.assert_allclose(calibrated.mu, 0.5 + 2.0 * bare.mu)
    np.testing.assert_allclose(calibrated.se, 3.0 * bare.se)
    np.testing.assert_allclose(calibrated.se_epi, 3.0 * bare.se_epi)
    np.testing.assert_allclose(calibrated.se_ale, 3.0 * bare.se_ale)
    seeds = [1, 2, 3]
    raw, _ = model.conditional_posterior_function_draw(
        x_whatif, y_whatif, Q, params=BARE, function_seeds=seeds
    )
    drawn, _ = model.conditional_posterior_function_draw(
        x_whatif, y_whatif, Q, params=CALIBRATED, function_seeds=seeds
    )
    want = (0.5 + 2.0 * bare.mu)[..., None] + 3.0 * (raw - bare.mu[..., None])
    np.testing.assert_allclose(drawn, want)


def test_calibrator_with_wrong_output_count_is_rejected(tmp_path) -> None:
    model = _simple_model(tmp_path)
    wrong = AffineCalibrator.identity(2)
    params = replace(BARE, calibrator=wrong)
    with pytest.raises(ValueError, match="calibrator has 2 outputs"):
        model.posterior(Q, params=params)
    with pytest.raises(ValueError, match="calibrator has 2 outputs"):
        model.batch_posterior(Q, paramss=[params])
    with pytest.raises(ValueError, match="calibrator has 2 outputs"):
        model.posterior_function_draw(Q, params=params, function_seeds=[1])


def test_calibrator_is_not_applied_twice_when_refitting(tmp_path) -> None:
    model = _simple_model(tmp_path)
    plain = fit_affine_calibrator(model, params=BARE, num_samples=20, seed=3)
    again = fit_affine_calibrator(model, params=CALIBRATED, num_samples=20, seed=3)
    np.testing.assert_array_equal(plain.a, again.a)
    np.testing.assert_array_equal(plain.b, again.b)
    np.testing.assert_array_equal(plain.c, again.c)


def test_ask_warm_start_does_not_reuse_the_calibrator(tmp_path) -> None:
    model = _simple_model(tmp_path)
    fitted = ENNStatefulFitter(3, seed=1).ask(
        model, num_fit_candidates=4, num_fit_samples=8, params_warm_start=CALIBRATED
    )
    assert fitted.calibrator is None


def test_adding_rows_keeps_the_calibrator_on_existing_params(tmp_path) -> None:
    model = _simple_model(tmp_path)
    params = ENNStatefulFitter(3, seed=1).ask(
        model, num_fit_candidates=4, num_fit_samples=8, affine_calibrate=True
    )
    calibrator = params.calibrator
    model.add(np.array([[0.2, 0.2]]), np.array([[1.4]]))
    assert params.calibrator is calibrator
    bare = model.posterior(Q, params=replace(params, calibrator=None))
    calibrated = model.posterior(Q, params=params)
    np.testing.assert_allclose(calibrated.se, calibrator.c * bare.se)


def test_calibrator_is_excluded_from_hash_and_equality() -> None:
    assert BARE == CALIBRATED
    assert hash(BARE) == hash(CALIBRATED)
