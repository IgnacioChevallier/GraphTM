#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./build_and_run.sh 7          # build with BOARD_DIM=7 and run
#   ./build_and_run.sh 3 4 5 6    # build and run for multiple sizes

if [ "$#" -eq 0 ]; then
  echo "Usage: $0 <BOARD_DIM> [<BOARD_DIM> ...]" >&2
  exit 1
fi

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

for dim in "$@"; do
  echo "==> Building with BOARD_DIM=${dim}"
  make clean >/dev/null 2>&1 || true
  make BOARD_DIM="${dim}"
  echo "==> Running ./hex (BOARD_DIM=${dim})"
  ./hex
done



