.PHONY: all clean test install rust-test python-test lint

# Default target: build the Rust extension in release mode
all:
	cd rust/crates/enn-py && maturin build --release

# Install both the Rust extension and Python package
install:
	@echo "Building and installing Rust extension (rust/crates/enn-py)..."
	cd rust/crates/enn-py && maturin develop --release
	@echo "Installing Python package (ennbo)..."
	pip install -e .
	@echo "Installation complete!"

# Run all tests (Rust and Python)
test: rust-test python-test

# Run Rust tests only
rust-test:
	cd rust && cargo test

# Run Python tests only
python-test:
	PYTHONPATH=src pytest -sv tests --tb=short

# Run linters
lint:
	cd rust && cargo clippy --all-targets --all-features -- -D warnings
	ruff check
	kiss check

# Clean build artifacts
clean:
	cd rust && cargo clean
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
