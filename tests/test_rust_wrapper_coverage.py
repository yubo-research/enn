from __future__ import annotations

import types

import numpy as np
import pytest


class _FakeInner:
    def __init__(self):
        self.last_tell = None
        self._tr_obs_count = 7
        self._tr_length = 0.33

    def telemetry(self):
        return types.SimpleNamespace(dt_fit=1.0, dt_gen=2.0, dt_sel=3.0, dt_tell=4.0)

    def init_progress(self):
        return (1, 4)

    def ask(self, num_arms, seed):
        assert seed >= 0
        return np.full((num_arms, 2), 0.5, dtype=float)

    def tell(self, x, y, seed):
        assert seed >= 0
        self.last_tell = (np.asarray(x), np.asarray(y))

    def tr_obs_count(self):
        return self._tr_obs_count

    def tr_length(self):
        return self._tr_length


def test_rust_optimizer_wrapper_methods():
    from enn.turbo.config import (
        AcqType,
        ENNFitConfig,
        ENNSurrogateConfig,
        turbo_enn_config,
    )
    from enn.turbo.rust_optimizer import RustOptimizer, is_rust_supported_config

    cfg = turbo_enn_config(
        acq_type=AcqType.UCB,
        enn=ENNSurrogateConfig(k=3, fit=ENNFitConfig(num_fit_samples=10)),
        num_init=2,
    )
    assert is_rust_supported_config(cfg)

    bounds = np.array([[-1.0, 1.0], [0.0, 2.0]], dtype=float)
    rng = np.random.default_rng(0)
    inner = _FakeInner()
    opt = RustOptimizer(bounds=bounds, config=cfg, rng=rng, inner=inner)

    x = opt.ask(3)
    assert x.shape == (3, 2)
    assert np.allclose(x[:, 0], 0.0)  # midpoint of [-1, 1]
    assert np.allclose(x[:, 1], 1.0)  # midpoint of [0, 2]
    assert opt.tr_obs_count == 7
    assert opt.tr_length == 0.33
    assert opt.init_progress == (1, 4)
    tele = opt.telemetry()
    assert (tele.dt_fit, tele.dt_gen, tele.dt_sel, tele.dt_tell) == (1.0, 2.0, 3.0, 4.0)

    y = np.array([1.0, 2.0, 3.0], dtype=float)
    y_out = opt.tell(x, y)
    assert y_out.shape == (3, 1)
    x_unit, y_seen = inner.last_tell
    assert np.all((0.0 <= x_unit) & (x_unit <= 1.0))
    assert y_seen.shape == (3, 1)


def test__rust_module_exports_optimizer_when_available():
    import enn._rust as rust_mod

    assert hasattr(rust_mod, "Optimizer")
    assert callable(rust_mod.sobol_sequence)
    assert callable(rust_mod.create_optimizer_enn)
    assert callable(rust_mod.create_optimizer_zero)
    assert callable(rust_mod.create_optimizer_lhd)
    opt = rust_mod.create_optimizer_zero(np.array([[0.0, 1.0], [0.0, 1.0]]), 2, 42)
    assert isinstance(opt, rust_mod.Optimizer)


def test_rust_optimizer_factory_rust_and_python_paths(monkeypatch):
    import enn.turbo.optimizer as py_opt
    import enn.turbo.rust_optimizer as ro
    from enn.turbo.config import (
        AcqType,
        ENNFitConfig,
        ENNSurrogateConfig,
        turbo_enn_config,
        turbo_one_config,
    )

    created = {}

    def _fake_create_optimizer_enn(bounds, k, num_init, seed, config_overrides=None):
        created["args"] = (np.asarray(bounds).shape, k, num_init, seed)
        return _FakeInner()

    monkeypatch.setattr(ro._rust, "create_optimizer_enn", _fake_create_optimizer_enn)

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    cfg_rust = turbo_enn_config(
        acq_type=AcqType.UCB,
        enn=ENNSurrogateConfig(k=5, fit=ENNFitConfig(num_fit_samples=10)),
        num_init=3,
    )
    out_rust = ro.create_optimizer(
        bounds=bounds, config=cfg_rust, rng=np.random.default_rng(0)
    )
    assert isinstance(out_rust, ro.RustOptimizer)
    assert created["args"][1] == 5
    assert created["args"][2] == 3

    sentinel = object()
    monkeypatch.setattr(py_opt, "create_optimizer", lambda **kwargs: sentinel)
    cfg_py = turbo_one_config(num_init=2)
    out_py = ro.create_optimizer(
        bounds=bounds, config=cfg_py, rng=np.random.default_rng(0)
    )
    assert out_py is sentinel


def test_rust_optimizer_factory_no_surrogate_path(monkeypatch):
    import enn.turbo.rust_optimizer as ro
    from enn.turbo.config import turbo_zero_config

    called = {}

    def _fake_create_optimizer_zero(bounds, num_init, seed, config_overrides=None):
        called["args"] = (np.asarray(bounds).shape, num_init, seed)
        return _FakeInner()

    monkeypatch.setattr(ro._rust, "create_optimizer_zero", _fake_create_optimizer_zero)

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    cfg = turbo_zero_config(num_init=4)
    out = ro.create_optimizer(bounds=bounds, config=cfg, rng=np.random.default_rng(1))
    assert isinstance(out, ro.RustOptimizer)
    assert called["args"][1] == 4


def test_rust_optimizer_factory_lhd_only_path(monkeypatch):
    import enn.turbo.rust_optimizer as ro
    from enn.turbo.config import lhd_only_config

    called = {}

    def _fake_create_optimizer_lhd(bounds, num_init, seed, config_overrides=None):
        called["args"] = (np.asarray(bounds).shape, num_init, seed)
        return _FakeInner()

    monkeypatch.setattr(ro._rust, "create_optimizer_lhd", _fake_create_optimizer_lhd)

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    cfg = lhd_only_config(num_init=6)
    out = ro.create_optimizer(bounds=bounds, config=cfg, rng=np.random.default_rng(2))
    assert isinstance(out, ro.RustOptimizer)
    assert called["args"][1] == 6


def test_rust_optimizer_factory_unsupported_surrogate_errors(monkeypatch):
    import enn.turbo.rust_optimizer as ro

    class _Cfg:
        def __init__(self):
            self.surrogate = object()
            self.init = types.SimpleNamespace(num_init=1, init_strategy=object())
            self.trust_region = object()

    monkeypatch.setattr(ro, "is_rust_supported_config", lambda _cfg: True)
    with pytest.raises(ValueError, match="Unsupported surrogate config"):
        ro.create_optimizer(
            bounds=np.array([[0.0, 1.0]], dtype=float),
            config=_Cfg(),
            rng=np.random.default_rng(0),
        )
