.PHONY: all install clean test build-ext rust-test python-test python-test-body lint wheels wheelsl \
	pypi-build pypi-publish pypi-auth-check

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
MATURIN_AUDITWHEEL := --auditwheel skip
# PyO3 `extension-module` omits libpython; macOS ld rejects undefined `_Py*` when nextest
# links ennbo-py as a cdylib. Linux GNU ld allows it. Scoped to rust-test only — maturin
# release builds must not inherit this (they use pyo3-build-config link args instead).
RUST_TEST_ENV := RUSTFLAGS="-C link-arg=-undefined -C link-arg=dynamic_lookup"
else
MATURIN_AUDITWHEEL :=
RUST_TEST_ENV :=
endif

# Default: build a release extension for the local platform.
all:
	maturin build --release $(MATURIN_AUDITWHEEL)

# Install the mixed Python/Rust package in editable mode (USearch always on; see pyproject [tool.maturin]).
# PIP_QUIET=1 drops pip's "Ignoring … markers 'extra == …'" INFO lines for unselected extras.
install:
	@echo "Building and installing Python/Rust package (see pyproject [tool.maturin])..."
	PIP_QUIET=1 maturin develop --release
	@echo "Installation complete!"

# Build the PyO3 extension into src/enn/ for PYTHONPATH=src pytest runs.
build-ext:
	PIP_QUIET=1 maturin develop --release

# Run all tests (Rust then Python; build-ext once — parallel rust-test + maturin races cargo).
test: build-ext rust-test python-test-body

# Run Rust tests only
rust-test:
	cd rust && $(RUST_TEST_ENV) cargo nextest run --test-threads=8

# Run Python tests only
python-test: build-ext python-test-body

python-test-body:
	PYTHONPATH=src pytest tests --tb=short -q

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
