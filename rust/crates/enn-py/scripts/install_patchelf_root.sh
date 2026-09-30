#!/usr/bin/env bash
set -eu

[ "$(uname -s)" = Linux ] || exit 0
[ "$(id -u)" -eq 0 ] || exit 0

min_maj=0
min_min=14
ver=0.18.0

patchelf_ok() {
  local line maj mi rest
  command -v patchelf >/dev/null 2>&1 || return 1
  line="$(patchelf --version 2>/dev/null | head -n1 || true)"
  case "$line" in
    patchelf\ *)
      rest="${line#patchelf }"
      maj="${rest%%.*}"
      rest="${rest#*.}"
      mi="${rest%%.*}"
      if [ "$maj" -gt "$min_maj" ]; then return 0; fi
      if [ "$maj" -eq "$min_maj" ] && [ "$mi" -ge "$min_min" ]; then return 0; fi
      ;;
  esac
  return 1
}

patchelf_ok && exit 0

arch=$(uname -m)
case "$arch" in
  x86_64)
    dist="patchelf-${ver}-x86_64.tar.gz"
    want_sha=ce84f2447fb7a8679e58bc54a20dc2b01b37b5802e12c57eece772a6f14bf3f0
    ;;
  aarch64)
    dist="patchelf-${ver}-aarch64.tar.gz"
    want_sha=ae13e2effe077e829be759182396b931d8f85cfb9cfe9d49385516ea367ef7b2
    ;;
  *)
    echo "install_patchelf_root.sh: unsupported arch $arch" >&2
    exit 1
    ;;
esac

url="https://github.com/NixOS/patchelf/releases/download/${ver}/${dist}"
cache="${XDG_CACHE_HOME:-/var/tmp}/enn-patchelf"
mkdir -p "$cache"
tmp="$cache/${dist}.part"
curl -fLsS --retry 3 -o "$tmp" "$url"
if command -v sha256sum >/dev/null 2>&1; then
  echo "$want_sha  $tmp" | sha256sum -c -
else
  got=$(shasum -a 256 "$tmp" | awk '{print $1}')
  if [ "$got" != "$want_sha" ]; then
    echo "install_patchelf_root.sh: sha256 mismatch for $dist" >&2
    exit 1
  fi
fi
td=$(mktemp -d)
tar -xzf "$tmp" -C "$td"
install -m0755 "$td/bin/patchelf" /usr/local/bin/patchelf
rm -rf "$td" "$tmp"
