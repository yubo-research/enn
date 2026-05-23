#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/rust"

require() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "error: missing required command: $1" >&2
    echo "Install sccache (e.g. brew install sccache) and ensure it is on PATH." >&2
    exit 1
  fi
}

require sccache
require cargo

sccache --zero-stats

echo "First build (populate sccache)..."
cargo build --workspace

echo "Clean and rebuild (expect cache hits)..."
cargo clean
cargo build --workspace

stats="$(sccache --show-stats)"
echo "$stats"

hits="$(echo "$stats" | awk '/^Cache hits[[:space:]]+[0-9]+[[:space:]]*$/ { print $3; exit }')"
if [[ "${hits:-0}" -lt 1 ]]; then
  echo "error: expected sccache cache hits after rebuild, got: ${hits:-0}" >&2
  exit 1
fi

echo "sccache OK (cache hits: ${hits})"
