from __future__ import annotations


def test_enn_package_reexports_are_importable():
    import enn.enn.enn as enn_mod

    assert enn_mod.DrawInternals is not None
    assert enn_mod.NeighborData is not None
    assert enn_mod.WeightedStats is not None
    assert set(enn_mod.__all__) >= {
        "DrawInternals",
        "NeighborData",
        "WeightedStats",
    }
