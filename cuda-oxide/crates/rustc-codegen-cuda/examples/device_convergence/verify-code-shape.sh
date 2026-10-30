#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# As in the other examples, smoketest.sh discovers this script automatically.
# Require the invariant summation loop to disappear from optimized PTX, then
# inspect pre-opt IR to preserve the collective/non-collective classification.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ll="${1:-${root}/device_convergence.ll}"
ptx="${2:-${root}/device_convergence.ptx}"

for artifact in "${ll}" "${ptx}"; do
    if [[ ! -s "${artifact}" ]]; then
        echo "error: missing ${artifact}; run cargo oxide build device_convergence" >&2
        exit 1
    fi
done

python3 - "${ll}" "${ptx}" <<'VERIFY_PY'
import re
import sys

ll_path, ptx_path = sys.argv[1:]
ll = re.sub(r";[^\n]*", "", open(ll_path, encoding="utf-8").read())
ptx = re.sub(r"/\*.*?\*/|//[^\n]*", "", open(ptx_path, encoding="utf-8").read(), flags=re.S)
groups = dict(re.findall(r"^attributes #(\d+) = \{([^}]*)\}", ll, re.M))


def convergent(line):
    """Resolve convergence from inline attributes or referenced attribute groups.
    Group numbers are deliberately not fixed to the exporter's current #0.
    """
    attributes = line + " " + " ".join(groups[n] for n in re.findall(r"#(\d+)", line))
    return re.search(r"\bconvergent\b", attributes) is not None


def definition(name):
    """Require exactly one definition with this unmangled helper or entry name.
    Missing helpers must fail instead of making an absence check pass vacuously.
    """
    matches = re.findall(r"^define [^\n]*@" + re.escape(name) + r"\([^\n]*", ll, re.M)
    if len(matches) != 1:
        raise SystemExit(f"error: expected one definition for {name}, got {len(matches)}")
    return matches[0]


def check_symbol(name, expected):
    """Check both the definition and every direct call to a named helper.
    Each helper must have callers so the test exercises propagation through edges.
    """
    lines = [definition(name)]
    calls = re.findall(r"^[^\n]*\bcall\b[^\n]*@" + re.escape(name) + r"\([^\n]*", ll, re.M)
    if not calls:
        raise SystemExit(f"error: no calls to {name}; helper chain disappeared")
    for line in lines + calls:
        if convergent(line) != expected:
            raise SystemExit(f"error: {name} expected convergent={expected}: {line.strip()}")


def entry_body(name):
    """Extract a PTX entry through its closing brace at column zero.
    Nested call-argument blocks are indented and do not terminate the entry.
    """
    match = re.search(r"\.entry\s+" + re.escape(name) + r"\([^{}]*\)\s*\{(.*?)^\}", ptx, re.S | re.M)
    if match is None:
        raise SystemExit(f"error: missing PTX entry body {name}")
    return match.group(1)


# Check PTX first so unchanged upstream fails on an observable code-generation
# difference even if its pre-opt convergence attributes are not inspected.
body = entry_body("loop_invariant_probe")
calls = body.count("device_convergence__thread_value_wrapper")
if calls != 1:
    raise SystemExit(f"error: loop_invariant_probe expected one invariant helper call, got {calls}")
labels = {m.group(1): m.start() for m in re.finditer(r"^\s*([\w.$]+):", body, re.M)}
backedges = [m.group(1) for m in re.finditer(r"\bbra(?:\.uni)?\s+([\w.$]+)\s*;", body)
             if m.group(1) in labels and labels[m.group(1)] < m.start()]
if backedges:
    raise SystemExit(f"error: loop_invariant_probe retains a PTX loop: {backedges}")
multiplies = len(re.findall(r"\bmul\.lo\.[su]32\b", body))
if multiplies != 1:
    raise SystemExit(f"error: loop_invariant_probe expected one invariant multiply, got {multiplies}")
print("loop_invariant_probe PTX: PASS (one helper call, one multiply, no loop)")

for helper in ("arithmetic_leaf", "arithmetic_wrapper", "thread_value_leaf", "thread_value_wrapper"):
    check_symbol("device_convergence__" + helper, False)
for helper in ("barrier_leaf", "barrier_wrapper", "shuffle_leaf", "shuffle_wrapper"):
    check_symbol("device_convergence__" + helper, True)
for entry, expected in (("arithmetic_probe", False), ("loop_invariant_probe", False),
                        ("barrier_probe", True), ("shuffle_probe", True)):
    if convergent(definition(entry)) != expected:
        raise SystemExit(f"error: {entry} expected convergent={expected}")

# Also require the actual collective seeds and their protection in the IR.
for intrinsic in (r"llvm\.nvvm\.barrier[^ (]*", r"llvm\.nvvm\.shfl[^ (]*"):
    calls = re.findall(r"^[^\n]*\bcall\b[^\n]*@" + intrinsic + r"\([^\n]*", ll, re.M)
    if not calls or not all(convergent(line) for line in calls):
        raise SystemExit(f"error: missing convergent intrinsic calls matching {intrinsic}")

for entry in ("arithmetic_probe", "loop_invariant_probe", "barrier_probe", "shuffle_probe"):
    if re.search(r"\.entry\s+" + re.escape(entry) + r"\(", ptx) is None:
        raise SystemExit(f"error: missing PTX entry {entry}")
for collective in (r"\bbar\.sync\b", r"\bshfl\.sync\.bfly\b"):
    if re.search(collective, ptx) is None:
        raise SystemExit(f"error: missing PTX collective {collective}")
print("device_convergence code shape: PASS")
VERIFY_PY
