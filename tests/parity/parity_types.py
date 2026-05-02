from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class ParityCase:
    name: str
    endpoint: str
    passed: bool
    error: str | None = None
    backend: str = "rust_vs_python"
    metrics: dict[str, float] | None = None


@dataclass
class ParityReport:
    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    rust_available: bool = False
    cases: list[ParityCase] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "rust_available": self.rust_available,
            "pct_parity": (self.passed / self.total * 100) if self.total > 0 else 0.0,
            "cases": [asdict(c) for c in self.cases],
        }
