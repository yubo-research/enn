.PHONY: all install clean test rust-test python-test lint wheels wheelsl \
	pypi-build pypi-publish pypi-auth-check

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
MATURIN_AUDITWHEEL := --auditwheel skip
else
MATURIN_AUDITWHEEL :=
endif

# Default: build a release extension for the local platform.
all:
	maturin build --release $(MATURIN_AUDITWHEEL)

# Install the mixed Python/Rust package in editable mode (USearch always on; see pyproject [tool.maturin]).
install:
	@echo "Building and installing Python/Rust package (see pyproject [tool.maturin])..."
	maturin develop --release --uv
	@echo "Installation complete!"

# Run all tests (Rust and Python)
test: rust-test python-test

# Run Rust tests only
rust-test:
	cd rust && cargo nextest run

# Run Python tests only
python-test:
	PYTHONPATH=src pytest -sv tests --tb=short

# Run linters
lint:
	cd rust && cargo clippy --all-targets --all-features -- -D warnings
	ruff check
	kiss check

# Build local PyPI wheel artifacts for the supported release tags.
wheels:
	scripts/build_wheels.sh

wheelsl: wheels

# --- PyPI (ennbo): token in MATURIN_PYPI_TOKEN, or credentials in ~/.pypirc ---
pypi-build:
	maturin build --release $(MATURIN_AUDITWHEEL)

# Note: `maturin publish` builds again before upload (same as a clean "build then publish").
pypi-publish:
	maturin publish --non-interactive

# Hits PyPI with your credentials but skips files already on the index (good auth smoke test).
pypi-auth-check: pypi-build
	maturin publish --non-interactive --skip-existing

# Clean build artifacts
clean:
	cd rust && cargo clean
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
