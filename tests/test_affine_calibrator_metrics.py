"""Metric gates and ablation matrix for AffineCalibrator (proposal-ols step 3)."""

from __future__ import annotations

import numpy as np

from enn import AffineCalibrator
from enn.enn.enn_normal import ENNNormal
from evals.flat_sphere import gaussian_loglik
from ops.qa import nrmse, rcorr


def _metrics(y: np.ndarray, mu: np.ndarray, se: np.ndarray) -> dict[str, float]:
    return {
        "nrmse": nrmse(y, mu),
        "loglik": gaussian_loglik(y, mu, se),
        "rcorr": rcorr(y, mu),
    }


def _ablate(
    raw: ENNNormal,
    *,
    mean_mode: str,
    se_mode: str,
    a_hat: np.ndarray,
    b_hat: np.ndarray,
    c_hat: np.ndarray,
) -> ENNNormal:
    """mean_mode in {raw, a_only, full}; se_mode in {raw, abs_b, c}."""
    m = int(raw.mu.shape[-1])
    if mean_mode == "raw":
        a = np.zeros(m)
        b = np.ones(m)
    elif mean_mode == "a_only":
        a = np.asarray(a_hat, dtype=float).ravel()
        b = np.ones(m)
    else:
        a = np.asarray(a_hat, dtype=float).ravel()
        b = np.asarray(b_hat, dtype=float).ravel()
    pad = (1,) * (raw.mu.ndim - 1) + (m,)
    mu = a.reshape(pad) + b.reshape(pad) * raw.mu
    if se_mode == "raw":
        scale = np.ones(m)
    elif se_mode == "abs_b":
        scale = np.abs(b)
    else:
        scale = np.asarray(c_hat, dtype=float).ravel()
    s = scale.reshape(pad)
    se_epi = s * raw.se_epi
    se_ale = s * raw.se_ale
    return ENNNormal(
        mu=mu,
        se=np.hypot(se_epi, se_ale),
        se_epi=se_epi,
        se_ale=se_ale,
        idx=raw.idx,
        y_bounds=raw.y_bounds,
    )


def _shrinkage_bundle(n: int = 80) -> tuple[np.ndarray, ENNNormal, np.ndarray, np.ndarray, np.ndarray]:
    y = np.linspace(-1.0, 1.0, n).reshape(-1, 1)
    mu = 0.5 * y
    se = np.full_like(mu, float(np.sqrt(np.mean((y - mu) ** 2))))
    raw = ENNNormal(mu=mu, se=se, se_epi=se.copy(), se_ale=np.zeros_like(se))
    cal = AffineCalibrator.identity(1)
    a_hat, b_hat = cal.fit(mu, y)
    c_hat = cal.fit_residual_scale(mu, se, y)
    return y, raw, a_hat, b_hat, c_hat


def test_metric_gates_before_after_full_affine_c():
    """nRMSE/loglik/rcorr before vs after full affine + residual c on shrinkage toy."""
    y, raw, a_hat, b_hat, c_hat = _shrinkage_bundle()
    before = _metrics(y, raw.mu, raw.se)
    out = _ablate(
        raw,
        mean_mode="full",
        se_mode="c",
        a_hat=a_hat,
        b_hat=b_hat,
        c_hat=c_hat,
    )
    after = _metrics(y, out.mu, out.se)
    assert after["nrmse"] < before["nrmse"]
    assert after["loglik"] > before["loglik"]
    assert np.isfinite(before["rcorr"]) and np.isfinite(after["rcorr"])
    assert after["rcorr"] >= before["rcorr"] - 1e-12
    np.testing.assert_allclose(out.se, np.hypot(out.se_epi, out.se_ale))


def test_ablation_matrix_mean_times_se():
    """Ablation: raw / a-only / full × {raw SE, |b| SE, c SE}; hypot when c used."""
    y, raw, a_hat, b_hat, c_hat = _shrinkage_bundle()
    rows = {}
    for mean_mode in ("raw", "a_only", "full"):
        for se_mode in ("raw", "abs_b", "c"):
            pred = _ablate(
                raw,
                mean_mode=mean_mode,
                se_mode=se_mode,
                a_hat=a_hat,
                b_hat=b_hat,
                c_hat=c_hat,
            )
            rows[(mean_mode, se_mode)] = _metrics(y, pred.mu, pred.se)
            if se_mode == "c":
                np.testing.assert_allclose(
                    pred.se, np.hypot(pred.se_epi, pred.se_ale), rtol=1e-10
                )
    assert rows[("full", "c")]["nrmse"] < rows[("raw", "raw")]["nrmse"]
    assert rows[("full", "c")]["loglik"] > rows[("raw", "raw")]["loglik"]
    assert rows[("full", "c")]["loglik"] > rows[("full", "abs_b")]["loglik"]
    assert rows[("full", "c")]["nrmse"] < rows[("a_only", "c")]["nrmse"]
    assert len(rows) == 9
