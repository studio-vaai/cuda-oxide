#!/usr/bin/env bash
# Pass/fail check that every `v_*` (repr(simd) vector) kernel in
# simd_vec_ops.ptx coalesced its whole-vector load/store into wide transactions
# instead of N scalar loads. The `scalar_f32x4` (repr(C) array wrapper) kernel
# is the contrast and is expected to stay scalar.
#
# A vector kernel PASSES if its data-load count is fewer than its lane count
# (i.e. the N lanes were fused into 1-2 wide `ld.v2`/`ld.v4`/`ld.b{32,64}`
# transactions). Exit code = number of vector kernels that failed to coalesce.
#
# Usage:
#   cargo oxide build simd_vec_ops      # emits simd_vec_ops.ptx next to this
#   ./analyze.sh                        # or: ./analyze.sh path/to/file.ptx
set -uo pipefail

PTX="${1:-$(dirname "$0")/simd_vec_ops.ptx}"
if [[ ! -f "$PTX" ]]; then
  echo "PTX not found: $PTX (run 'cargo oxide build simd_vec_ops' first)" >&2
  exit 2
fi

fails=0
printf '%-14s %-5s %-6s %-16s %s\n' KERNEL LANES LOADS MNEMONIC RESULT
printf '%-14s %-5s %-6s %-16s %s\n' "------" "-----" "-----" "--------" "------"

for k in $(grep -oE '^\.visible \.entry [A-Za-z0-9_]+' "$PTX" | awk '{print $3}'); do
  body=$(awk -v k="$k" '$0 ~ ("^\\.visible \\.entry "k"\\(") {f=1} f&&/^\}/{exit} f{print}' "$PTX")
  loads=$(printf '%s\n' "$body" | grep -oE 'ld\.[a-z0-9.]+' | grep -v 'ld\.param')
  nload=$(printf '%s\n' "$loads" | grep -c .)
  mnem=$(printf '%s\n' "$loads" | sort -u | tr '\n' ',' | sed 's/,$//')
  lanes=$(printf '%s' "$k" | grep -oE 'x[0-9]+$' | tr -d x)

  if [[ "$k" == scalar_* ]]; then
    result="scalar (expected contrast)"
  elif [[ -n "$lanes" && "$nload" -lt "$lanes" ]]; then
    result="PASS"
  else
    result="FAIL (not coalesced)"
    fails=$((fails + 1))
  fi
  printf '%-14s %-5s %-6s %-16s %s\n' "$k" "${lanes:-?}" "$nload" "$mnem" "$result"
done

echo
if [[ "$fails" -eq 0 ]]; then
  echo "✓ all vector kernels coalesced into wide transactions"
else
  echo "✗ $fails vector kernel(s) did NOT coalesce"
fi
exit "$fails"
