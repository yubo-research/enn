# Testing `enn-py`

`enn-py` is a Python extension crate (PyO3 `cdylib`).
The reliable test path is Python-side after installing the extension.

## Why not `cargo test -p enn-py` for wrapper behavior?

On some systems, Rust test binaries for PyO3 crates fail to link Python C symbols.
That is an embedding/link mode issue, not a missing algorithm implementation.

## Recommended workflow

From repo root:

1. Install the extension into the active Python env:

```bash
cd rust/crates/enn-py
maturin develop
```

2. Run Python parity/contract tests:

```bash
cd /path/to/repo
PYTHONPATH=src:$PYTHONPATH python -m pytest tests/python_api -q
PYTHONPATH=src:$PYTHONPATH python -m pytest tests/parity/helpers -q
```

## Rust-side checks that should still pass

```bash
cd /path/to/repo/rust
cargo test
cargo clippy --all-targets --all-features -- -D warnings
```
