.PHONY: all install clean test build-ext rust-test python-test python-test-body python-slow-test lint wheels wheelsl \
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
install:
	@echo "Building and installing Python/Rust package (see pyproject [tool.maturin])..."
	maturin develop --release
	@echo "Installation complete!"

# Build the PyO3 extension into src/enn/ for PYTHONPATH=src pytest runs.
build-ext:
	maturin develop --release

# Run all tests (Rust then Python; build-ext once — parallel rust-test + maturin races cargo).
test: build-ext rust-test python-test-body

# Run Rust tests only
rust-test:
	cd rust && $(RUST_TEST_ENV) cargo nextest run --test-threads=8

# Run Python tests only (fast gate: skips modules collected only for slow/integration coverage).
PYTHON_FAST_PLUGINS = \
	-p no:nbmake -p no:hypothesis -p no:aiohttp -p no:examples
PYTHON_SLOW_IGNORE = \
	--ignore=tests/test_turbo_gp.py \
	--ignore=tests/test_turbo_adversarial.py \
	--ignore=tests/test_ops_stress.py \
	--ignore=tests/test_ops_disk_rss_stress.py \
	--ignore=tests/test_disk_hnsw_background_flush.py \
	--ignore=tests/test_enn_index_driver.py \
	--ignore=tests/test_kiss_coverage.py \
	--ignore=tests/test_kiss_fullrepo_symbol_registry.py \
	--ignore=tests/parity \
	--ignore=tests/test_compare_ennbo_versions.py \
	--ignore=tests/test_notebooks.py \
	--ignore=tests/test_turbo_invariance.py \
	--ignore=tests/test_turbo_optimizer.py \
	--ignore=tests/test_components.py \
	--ignore=tests/test_enn_perf.py \
	--ignore=tests/test_python_fallback_surface.py \
	--ignore=tests/test_morbo_turbo_one.py \
	--ignore=tests/test_try_hnsw_disk.py \
	--ignore=tests/test_turbo_utils_funcs.py \
	--ignore=tests/test_turbo_optimizer_utils.py \
	--ignore=tests/test_enn_fit.py \
	--ignore=tests/test_candidate_gen_stats.py \
	--ignore=tests/test_morbo_separable_unimodal.py \
	--ignore=tests/test_enn_batch_posterior_train_regression.py \
	--ignore=tests/test_weak_mathy_golden.py \
	--ignore=tests/test_fallback_registry.py \
	--ignore=tests/test_turbo_strategies.py \
	--ignore=tests/python_api/test_create_optimizer_contract.py \
	--ignore=tests/test_optimizer_generate_smoke.py \
	--ignore=tests/test_tr_helpers_direct.py \
	--ignore=tests/test_raasp_candidates.py \
	--ignore=tests/test_trust_region.py \
	--ignore=tests/test_turbo_tr.py \
	--ignore=tests/test_protocols.py \
	--ignore=tests/test_incumbent_tracker.py \
	--ignore=tests/test_morbo_tr_direct.py \
	--ignore=tests/test_incumbent_selector.py \
	--ignore=tests/test_impl_helpers.py \
	--ignore=tests/test_rust_optimizer_kiss_tokens.py \
	--ignore=tests/test_rust_wrapper_coverage.py
python-test: build-ext python-test-body

python-test-body:
	PYTHONPATH=src pytest tests --tb=short -m "not slow" -q $(PYTHON_FAST_PLUGINS) $(PYTHON_SLOW_IGNORE)

# Slow/integration Python tests (not part of the default gate).
python-slow-test:
	PYTHONPATH=src pytest tests --tb=short -m "slow" -q
	PYTHONPATH=src pytest tests/test_turbo_gp.py tests/test_turbo_adversarial.py \
		tests/test_ops_stress.py tests/test_ops_disk_rss_stress.py \
		tests/test_disk_hnsw_background_flush.py tests/test_enn_index_driver.py \
		tests/test_kiss_coverage.py tests/test_kiss_fullrepo_symbol_registry.py \
		tests/parity tests/test_compare_ennbo_versions.py tests/test_notebooks.py \
		tests/test_turbo_invariance.py tests/test_turbo_optimizer.py tests/test_components.py \
		tests/test_enn_perf.py tests/test_python_fallback_surface.py \
		tests/test_morbo_turbo_one.py tests/test_try_hnsw_disk.py \
		tests/test_turbo_utils_funcs.py tests/test_turbo_optimizer_utils.py \
		tests/test_enn_fit.py tests/test_candidate_gen_stats.py \
		tests/test_morbo_separable_unimodal.py \
		tests/test_enn_batch_posterior_train_regression.py tests/test_weak_mathy_golden.py \
		tests/test_fallback_registry.py tests/test_turbo_strategies.py \
		tests/python_api/test_create_optimizer_contract.py \
		tests/test_optimizer_generate_smoke.py tests/test_tr_helpers_direct.py \
		tests/test_raasp_candidates.py tests/test_trust_region.py \
		tests/test_turbo_tr.py tests/test_protocols.py \
		tests/test_incumbent_tracker.py tests/test_morbo_tr_direct.py \
		tests/test_incumbent_selector.py tests/test_impl_helpers.py \
		tests/test_rust_optimizer_kiss_tokens.py tests/test_rust_wrapper_coverage.py \
		--tb=short -q

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
