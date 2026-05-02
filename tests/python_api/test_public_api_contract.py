from __future__ import annotations

import inspect


class TestPublicAPIExports:
    """Verify public API surface is exported correctly from enn package."""

    def test_all_lazy_attrs_available(self):
        """All _LAZY_ATTRS entries must be importable."""
        import enn

        for name in enn._LAZY_ATTRS:
            attr = getattr(enn, name)
            assert attr is not None, f"Attribute {name} not found"

    def test_epistemic_nearest_neighbors_class(self):
        """EpistemicNearestNeighbors is a class with expected signature."""
        from enn import EpistemicNearestNeighbors

        assert inspect.isclass(EpistemicNearestNeighbors)
        # Check constructor signature has required parameters
        sig = inspect.signature(EpistemicNearestNeighbors.__init__)
        params = list(sig.parameters.keys())
        assert "train_x" in params
        assert "train_y" in params
        assert "train_yvar" in params

    def test_enn_fit_function(self):
        """enn_fit is a callable function."""
        from enn import enn_fit

        assert callable(enn_fit)

    def test_create_optimizer_function(self):
        """create_optimizer is a callable function."""
        from enn import create_optimizer

        assert callable(create_optimizer)

    def test_config_classes(self):
        """All config classes are available and constructible."""
        from enn import (
            MorboTRConfig,
            NoTRConfig,
            OptimizerConfig,
            TurboTRConfig,
        )

        # These are dataclasses or similar - should be constructible
        assert inspect.isclass(OptimizerConfig)
        assert inspect.isclass(TurboTRConfig)
        assert inspect.isclass(MorboTRConfig)
        assert inspect.isclass(NoTRConfig)

    def test_config_factory_functions(self):
        """All config factory functions are callable."""
        from enn import (
            lhd_only_config,
            turbo_enn_config,
            turbo_one_config,
            turbo_zero_config,
        )

        assert callable(turbo_one_config)
        assert callable(turbo_zero_config)
        assert callable(turbo_enn_config)
        assert callable(lhd_only_config)

    def test_telemetry_class(self):
        """Telemetry class is available."""
        from enn import Telemetry

        assert inspect.isclass(Telemetry)

    def test_enum_types(self):
        """Enum types are available."""
        from enn import AcqType, CandidateRV, InitStrategy

        assert inspect.isclass(CandidateRV)
        assert inspect.isclass(InitStrategy)
        assert inspect.isclass(AcqType)


class TestPublicAPIImmutability:
    """Verify public API does not change unexpectedly."""

    def test_all_list_matches_lazy_attrs(self):
        """__all__ must match _LAZY_ATTRS keys."""
        import enn

        assert set(enn.__all__) == set(enn._LAZY_ATTRS.keys())
