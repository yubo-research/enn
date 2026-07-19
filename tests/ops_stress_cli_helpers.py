from __future__ import annotations

from click.testing import CliRunner


def assert_stress_cli_rejects(args: list[str], fragment: str) -> None:
    """Shared CliRunner reject assertion for ops/stress.py commands."""
    from ops.stress import cli

    result = CliRunner().invoke(cli, args)
    assert result.exit_code != 0
    assert fragment in result.output
