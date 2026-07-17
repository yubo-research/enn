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
    assert "structured_build_row_limit" in text
    assert "search_beam_width" in text
    assert "10000" in text
    assert "1000" in text
    assert "1024" in text


def test_ensure_config_file_creates_defaults(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    assert not cfg.exists()
    set_config_path(str(cfg))
    try:
        _assert_default_config_written(cfg)
    finally:
        set_config_path(None)
