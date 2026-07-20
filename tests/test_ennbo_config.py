from __future__ import annotations

from pathlib import Path

from enn._rust import ensure_config_file, set_config_path


def _assert_default_config_written(cfg: Path) -> None:
    path = ensure_config_file()
    assert Path(path) == cfg
    assert cfg.is_file()
    text = cfg.read_text(encoding="utf-8")
    assert "[bpann]" in text
    assert "index_compact_rows_per_fragment" in text
    assert "pending_flush_threshold" in text
    assert "pending_hard_flush_threshold" in text
    assert "structured_build_row_limit" in text
    assert "search_beam_width" in text
    assert "exhaustive_search_row_limit" in text
    assert "skip_refinement_row_limit" in text
    assert "search_fragment_budget_max" in text
    assert "index_compact_rows_per_fragment = 10000" in text
    assert "pending_flush_threshold = 250" in text
    assert "pending_hard_flush_threshold = 3000" in text
    assert "structured_build_row_limit = 1" in text
    assert "search_beam_width = 1" in text
    assert "exhaustive_search_row_limit = 2500" in text
    assert "skip_refinement_row_limit = 150000" in text
    assert "search_fragment_budget_max = 1" in text


def test_ensure_config_file_creates_defaults(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    assert not cfg.exists()
    set_config_path(str(cfg))
    try:
        _assert_default_config_written(cfg)
    finally:
        set_config_path(None)


def test_missing_hard_key_with_elevated_soft_preserves_soft(tmp_path: Path) -> None:
    """Absent hard must not wipe soft>1000 via full-default-fallback (Q5)."""
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        "[bpann]\npending_flush_threshold = 2000\nsearch_beam_width = 1\n",
        encoding="utf-8",
    )
    set_config_path(str(cfg))
    try:
        path = ensure_config_file()
    finally:
        set_config_path(None)
    assert Path(path) == cfg
    text = cfg.read_text(encoding="utf-8")
    assert "pending_flush_threshold = 2000" in text
    assert "pending_hard_flush_threshold" not in text
