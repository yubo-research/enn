#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LINUX_POLICIES=("2_31" "2_25")

have() {
  command -v "$1" >/dev/null 2>&1
}

require() {
  if ! have "$1"; then
    echo "error: missing required command: $1" >&2
    exit 1
  fi
}

check_wheel() {
  local directory="$1"
  local tag="$2"

  if ! compgen -G "${directory}/*${tag}.whl" >/dev/null; then
    echo "error: missing wheel with tag ${tag} in ${directory}" >&2
    exit 1
  fi
}

macos_tag_for_python311() {
  python3.11 -m pip debug --verbose 2>/dev/null \
    | awk '/cp311-cp311-macosx_.*_arm64/ && !seen { print $1; seen = 1 }'
}

deployment_target_from_macos_tag() {
  local tag="$1"
  python3.11 - "$tag" <<'PY'
import re
import sys

match = re.search(r"macosx_(\d+)_(\d+)_arm64$", sys.argv[1])
if not match:
    raise SystemExit(f"cannot parse macOS wheel tag: {sys.argv[1]}")
print(f"{match.group(1)}.{match.group(2)}")
PY
}

build_macos_arm64_cp311() {
  if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "Skipping macOS wheel: this host is not macOS."
    return
  fi
  if [[ "$(uname -m)" != "arm64" ]]; then
    echo "Skipping macOS arm64 wheel: this host is $(uname -m), not arm64."
    return
  fi

  require brew
  require python3.11

  local mac_tag
  local deployment_target

  mac_tag="$(macos_tag_for_python311)"
  if [[ -z "$mac_tag" ]]; then
    echo "error: could not find a compatible cp311 macOS arm64 tag" >&2
    exit 1
  fi
  deployment_target="$(deployment_target_from_macos_tag "$mac_tag")"

  echo "Building ${mac_tag}..."
  brew list faiss >/dev/null 2>&1 || brew install faiss
  brew list libomp >/dev/null 2>&1 || brew install libomp
  brew list openblas >/dev/null 2>&1 || brew install openblas
  brew list pkg-config >/dev/null 2>&1 || brew install pkg-config

  python3.11 -m pip install --upgrade pip maturin delocate

  export MACOSX_DEPLOYMENT_TARGET="$deployment_target"
  export CPPFLAGS="-I/opt/homebrew/include"
  export LDFLAGS="-L/opt/homebrew/lib"
  export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig"
  export DYLD_FALLBACK_LIBRARY_PATH="/opt/homebrew/lib"

  rm -rf dist/raw dist/repaired
  maturin build \
    --release \
    --interpreter python3.11 \
    --auditwheel skip \
    --out dist/raw
  MACOSX_DEPLOYMENT_TARGET="$deployment_target" \
    delocate-wheel -w dist/repaired -v dist/raw/*.whl
  check_wheel "dist/repaired" "$mac_tag"
}

build_linux_x86_64_cp312() {
  require docker

  for policy in "${LINUX_POLICIES[@]}"; do
    local tag="cp312-cp312-manylinux_${policy}_x86_64"
    local repair_policy="$policy"
    if [[ "$policy" == "2_25" ]]; then
      repair_policy="2_31"
    fi
    echo "Building ${tag}..."

    docker run --rm \
      --platform linux/amd64 \
      -e POLICY="$policy" \
      -e REPAIR_POLICY="$repair_policy" \
        -e CMAKE_INSTALL_LIBDIR=lib \
      -e HOST_UID="$(id -u)" \
      -e HOST_GID="$(id -g)" \
      -v "$ROOT":/io \
      -w /io \
      quay.io/pypa/manylinux2014_x86_64 \
      bash -lc '
        set -euxo pipefail
        curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs \
          | sh -s -- -y --profile minimal
        . "$HOME/.cargo/env"
        yum install -y blas-devel lapack-devel openblas-devel
        /opt/python/cp312-cp312/bin/python -m pip install --upgrade pip maturin
        rm -rf /io/rust/target/release/build/faiss-sys-* \
          /io/rust/target/release/deps/*faiss*
        build_wheel() {
          /opt/python/cp312-cp312/bin/maturin build \
            --release \
            --interpreter /opt/python/cp312-cp312/bin/python \
            --manylinux "$REPAIR_POLICY" \
            --auditwheel repair \
            --out "/io/dist/manylinux_${POLICY}"
        }
        repair_faiss_libdir() {
          for out in /io/rust/target/release/build/faiss-sys-*/out; do
            [ -d "$out/lib64" ] || continue
            [ -e "$out/lib" ] || ln -s lib64 "$out/lib"
          done
          rm -f /io/rust/target/release/deps/*faiss*
        }
        build_wheel || {
          repair_faiss_libdir
          build_wheel
        }
        if [ "$POLICY" != "$REPAIR_POLICY" ]; then
          /opt/python/cp312-cp312/bin/python -m pip install wheel
          /opt/python/cp312-cp312/bin/python -m wheel tags \
            --remove \
            --platform-tag "manylinux_${POLICY}_x86_64" \
            /io/dist/manylinux_${POLICY}/*"manylinux_${REPAIR_POLICY}_x86_64.whl"
        fi
        chown -R "${HOST_UID}:${HOST_GID}" /io/dist
      '

    check_wheel "dist/manylinux_${policy}" "$tag"
  done
}

build_macos_arm64_cp311
build_linux_x86_64_cp312

echo "Wheel builds complete. Artifacts are under dist/."
