#!/usr/bin/env bash
# Analyze cusimd_ops.ptx: per-kernel data load/store widths + a check for any
# vectorized / global-state-space memory instructions.
#
# Usage:
#   cargo oxide build cusimd_ops      # emits cusimd_ops.ptx next to this script
#   ./analyze.sh                      # or: ./analyze.sh path/to/cusimd_ops.ptx
set -euo pipefail

PTX="${1:-$(dirname "$0")/cusimd_ops.ptx}"
if [[ ! -f "$PTX" ]]; then
  echo "PTX not found: $PTX (run 'cargo oxide build cusimd_ops' first)" >&2
  exit 1
fi

echo "== Vectorized / global / non-coherent memory ops present? =="
n=$(grep -cE '\.global|\.v2|\.v4|\.nc' "$PTX" || true)
echo "  matches for .global/.v2/.v4/.nc : $n   (0 => none; CuSimd did NOT vectorize)"
echo

echo "== Per-kernel data load/store widths (ld.param/st.param excluded) =="
awk '
  /^\.visible \.entry/ { name=$3; sub(/\(.*/,"",name); k=name }
  {
    if ($0 ~ /(ld|st)\.b[0-9]+/ && $0 !~ /\.param/) {
      match($0, /(ld|st)\.b[0-9]+/); m=substr($0,RSTART,RLENGTH); c[k","m]++
    }
  }
  END { for (key in c) print key, c[key] }
' "$PTX" | sort
