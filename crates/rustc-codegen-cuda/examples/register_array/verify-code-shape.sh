#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

root="$(cd "$(dirname "$0")" && pwd)"
ptx="${1:-$root/register_array.ptx}"
test -s "$ptx"

for size in 3 4 16; do
    symbol="select_${size}"
    body="$(awk -v symbol="$symbol" '
        /\.visible[[:space:]]+\.entry/ && index($0, symbol "(") { inside = 1 }
        inside {
            print
            opens += gsub(/\{/, "{")
            closes += gsub(/\}/, "}")
            if (opens > 0 && opens == closes) exit
        }
    ' "$ptx")"
    if [[ -z "$body" ]]; then
        echo "entry not found: $symbol" >&2
        exit 1
    fi
    if grep -Eq '(^|[[:space:]])\.local|ld\.local|st\.local|cvta\.local|call([.]|[[:space:]])' <<<"$body"; then
        echo "unexpected local memory or outlined call in $symbol" >&2
        printf '%s\n' "$body" >&2
        exit 1
    fi
done

echo "register_array PTX shape: PASS (3/4/16 elements, no local memory or calls)"
