// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Advisory cold-link scheduling from the NVVM text we already generated or read.
//! Shared helper units contain unrelated functions, so input bytes overestimate
//! some kernels and underestimate deeply inlined ones. No compiler input, option,
//! cache identity, or output depends on these estimates.
use std::collections::{BTreeMap, BTreeSet};

const MAX_EXPANDED: u64 = 64_000_000;

#[derive(Default)]
struct Function {
    instructions: u64,
    calls: BTreeMap<String, u64>,
    noinline: bool,
}

#[derive(Default)]
pub(crate) struct Unit {
    functions: BTreeMap<String, Function>,
}

impl Unit {
    pub(crate) fn from_ir(ir: &[u8]) -> Self {
        let Ok(ir) = std::str::from_utf8(ir) else {
            return Self::default();
        };
        let mut result = Self::default();
        let mut current = None;
        for line in ir.lines().map(str::trim) {
            if line.starts_with("define ") {
                current = symbol(line).inspect(|name| {
                    result.functions.insert(
                        name.clone(),
                        Function {
                            noinline: line.split_whitespace().any(|token| token == "noinline"),
                            ..Function::default()
                        },
                    );
                });
                continue;
            }
            if line == "}" {
                current = None;
                continue;
            }
            let Some(function) = current
                .as_ref()
                .and_then(|name| result.functions.get_mut(name))
            else {
                continue;
            };
            if line.starts_with('%')
                || ["store ", "br ", "ret ", "call ", "switch ", "unreachable"]
                    .iter()
                    .any(|prefix| line.starts_with(prefix))
            {
                function.instructions += 1;
            }
            let call = line
                .split_once("call ")
                .or_else(|| line.split_once("invoke "))
                .map(|(_, call)| call);
            if let Some(call) = call {
                // Inline assembly and indirect calls have no direct callee.
                // In the latter case an @symbol may occur only in an argument.
                if let Some(at) = call.find('@')
                    && !call[..at].contains('(')
                    && !call[..at].contains(" asm ")
                    && let Some(callee) = symbol(call)
                {
                    *function.calls.entry(callee).or_default() += 1;
                }
            }
        }
        result
    }
}

/// Names from our exporter are legal identifiers. Also accept quoted LLVM
/// names, including escaped bytes, so unfamiliar text only loses an estimate.
fn symbol(text: &str) -> Option<String> {
    let text = text.split_once('@')?.1;
    if let Some(text) = text.strip_prefix('"') {
        let mut result = Vec::new();
        let mut bytes = text.bytes();
        while let Some(byte) = bytes.next() {
            match byte {
                b'"' => return String::from_utf8(result).ok(),
                b'\\' => {
                    let high = (bytes.next()? as char).to_digit(16)?;
                    let low = (bytes.next()? as char).to_digit(16)?;
                    result.push((high * 16 + low) as u8);
                }
                _ => result.push(byte),
            }
        }
        None
    } else {
        let end = text.find(|c: char| c == '(' || c.is_whitespace())?;
        (end != 0).then(|| text[..end].to_owned())
    }
}

/// Weight unique reachable instructions together with an estimate of repeated
/// inlining. Standalone noinline functions are counted once. Cycles terminate;
/// memoization keeps heavily branching call graphs linear in their edges.
pub(crate) fn estimate<'a>(
    units: impl IntoIterator<Item = &'a Unit>,
    kernels: &[String],
) -> Option<u64> {
    let mut functions = BTreeMap::new();
    for unit in units {
        functions.extend(
            unit.functions
                .iter()
                .map(|(name, function)| (name.as_str(), function)),
        );
    }
    if kernels
        .iter()
        .any(|name| !functions.contains_key(name.as_str()))
    {
        return None;
    }
    let mut pending: Vec<_> = kernels.iter().map(String::as_str).collect();
    let mut seen = BTreeSet::new();
    while let Some(name) = pending.pop() {
        if let Some(function) = functions.get(name)
            && seen.insert(name)
        {
            pending.extend(function.calls.keys().map(String::as_str));
        }
    }
    fn expanded<'a>(
        name: &'a str,
        functions: &BTreeMap<&'a str, &'a Function>,
        memo: &mut BTreeMap<&'a str, u64>,
        active: &mut BTreeSet<&'a str>,
    ) -> u64 {
        if active.contains(name) {
            return 0;
        }
        if active.len() >= 256 {
            return MAX_EXPANDED;
        }
        if let Some(cost) = memo.get(name) {
            return *cost;
        }
        let Some(function) = functions.get(name) else {
            return 0;
        };
        active.insert(name);
        let mut cost = function.instructions;
        for (callee, count) in &function.calls {
            if functions
                .get(callee.as_str())
                .is_some_and(|callee| !callee.noinline)
            {
                cost = cost
                    .saturating_add(count.saturating_mul(expanded(callee, functions, memo, active)))
                    .min(MAX_EXPANDED);
            }
        }
        active.remove(name);
        memo.insert(name, cost);
        cost
    }
    let selected: u64 = seen.iter().map(|name| functions[name].instructions).sum();
    let mut memo = BTreeMap::new();
    let mut active = BTreeSet::new();
    let mut inlined = 0u64;
    for name in &seen {
        if kernels.iter().any(|kernel| kernel.as_str() == *name) || functions[name].noinline {
            inlined = inlined.saturating_add(expanded(name, &functions, &mut memo, &mut active));
        }
    }
    // Integer equivalent of unique instructions + 0.1 * expanded instructions.
    Some(selected.saturating_mul(10).saturating_add(inlined))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unrelated_helper_definitions_do_not_change_kernel_priority() {
        let ir = b"define void @kernel() {\ncall void @helper()\nret void\n}\n\
            define void @helper() {\n%a = add i32 1, 2\nret void\n}\n";
        let original = Unit::from_ir(ir);
        let mut larger = ir.to_vec();
        larger.extend_from_slice(b"define void @unused() {\n");
        for _ in 0..1000 {
            larger.extend_from_slice(b"%b = add i32 3, 4\n");
        }
        larger.extend_from_slice(b"ret void\n}\n");
        assert_eq!(estimate([&original], &["kernel".into()]), Some(44));
        assert_eq!(
            estimate([&Unit::from_ir(&larger)], &["kernel".into()]),
            Some(44)
        );
    }

    #[test]
    fn repeated_inline_calls_and_noinline_functions_have_different_costs() {
        let ir = "define void @kernel() {\ncall void @helper()\ncall void @helper()\nret void\n}\n\
            define void @helper() {\n%a = add i32 1, 2\nret void\n}\n";
        assert_eq!(
            estimate([&Unit::from_ir(ir.as_bytes())], &["kernel".into()]),
            Some(57)
        );
        let noinline = ir.replace("@helper() {", "@helper() noinline {");
        assert_eq!(
            estimate([&Unit::from_ir(noinline.as_bytes())], &["kernel".into()]),
            Some(55)
        );
    }

    #[test]
    fn recursive_and_exponentially_branching_graphs_are_bounded() {
        let recursive =
            Unit::from_ir(b"define void @kernel() {\ncall void @kernel()\nret void\n}\n");
        assert_eq!(estimate([&recursive], &["kernel".into()]), Some(22));
        let mut ir = String::new();
        for i in 0..70 {
            ir.push_str(&format!(
                "define void @f{i}() {{\ncall void @f{}()\ncall void @f{}()\nret void\n}}\n",
                i + 1,
                i + 1
            ));
        }
        let unit = Unit::from_ir(ir.as_bytes());
        assert_eq!(estimate([&unit], &["f0".into()]), Some(2100 + MAX_EXPANDED));
    }

    #[test]
    fn quoted_names_resolve_and_assembly_and_indirect_arguments_are_not_callees() {
        let unit = Unit::from_ir(b"define void @\"ker\\6Eel\"() {\ncall void asm \"@helper(\", \"\"()\ncall void %callee(ptr @helper)\nret void\n}\n\
            define void @helper() {\nret void\n}\n");
        assert_eq!(estimate([&unit], &["kernel".into()]), Some(33));
        assert_eq!(estimate([&unit], &["missing".into()]), None);
        assert_eq!(
            estimate([&Unit::from_ir(b"\xff")], &["kernel".into()]),
            None
        );
    }
}
