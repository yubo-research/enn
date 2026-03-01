#!/usr/bin/env bash
set -euo pipefail

# Run from repo root so paths from git status are correct
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "Scanning for untracked or modified Python files (git status)..."

status_output="$(git status --porcelain --untracked-files=all)"

if [[ -z "$status_output" ]]; then
  echo "Working tree clean. No forgotten .py files."
  exit 0
fi

# Select only:
# - untracked files ("??")
# - or files with unstaged changes (second status column non-space)
# We parse the raw porcelain line as:
#   XY <space> PATH
# where X is index status, Y is work-tree status.
py_files="$(
  printf '%s\n' "$status_output" |
    awk '{
      line = $0
      x = substr(line, 1, 1)
      y = substr(line, 2, 1)
      path = substr(line, 4)
      if ((x == "?" && y == "?") || y != " ")
        print path
    }' |
    grep -E '\.py$' || true
)"

if [[ -z "$py_files" ]]; then
  echo "No forgotten .py files detected."
  exit 0
fi

echo "Potential forgotten Python files:"
printf "%s\n" "$py_files"

# In local development, report but do not block. CI keeps this as a hard gate.
if [[ "${CI:-}" == "true" ]]; then
  exit 1
fi
echo "Non-CI run: reporting only."
exit 0
