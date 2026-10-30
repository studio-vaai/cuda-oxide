/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Read-only convergence analysis of the complete lowered LLVM call graph.
//!
//! Explicit LLVM attributes and primitive operation semantics seed the graph.
//! The immutable result is consumed by every definition and call emitter.

/// Core LLVM intrinsic classification for standard scalar and memory operations.
/// Unreviewed names remain unknown and therefore protected.
mod intrinsics;
/// Regressions for collective and thread-local intrinsic classification.
/// Unknown names must preserve the conservative external-call default.
#[cfg(test)]
mod tests;

use pliron::{
    builtin::{
        op_interfaces::{CallOpCallable, CallOpInterface, SymbolOpInterface},
        ops::ModuleOp,
    },
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    op::Op,
    operation::Operation,
};
use rustc_hash::FxHashMap;

use self::intrinsics::intrinsic_convergence;
use super::{externs::DeviceExternDecl, function::exported_function_name};
use crate::ops::{self, LlvmAttrValue, LlvmAttributesAttr};

/// Completed convergence requirements for definitions and referenced callees.
/// The map includes proven non-convergent local definitions; absent names are external.
#[derive(Default)]
pub(super) struct ConvergenceAnalysis {
    /// Final emitted symbol names mapped to their transitive convergence requirement.
    /// Unknown external targets are protected unless intrinsic semantics prove otherwise.
    functions: FxHashMap<String, bool>,
}

impl ConvergenceAnalysis {
    /// Query a definition or direct callee without changing the completed analysis.
    /// Names not visited during analysis use the same conservative intrinsic policy.
    pub(super) fn function_is_convergent(&self, name: &str) -> bool {
        self.functions
            .get(name)
            .copied()
            .unwrap_or_else(|| external_is_convergent(name))
    }

    /// Query the target's inferred requirement without inspecting call attributes.
    /// An indirect target is unknown and therefore remains protected.
    pub(super) fn callee_is_convergent(&self, callee: Option<&str>) -> bool {
        callee.is_none_or(|name| self.function_is_convergent(name))
    }
}

/// Classify a declaration using the single intrinsic policy, with unknowns protected.
/// Neither purity nor read-only memory effects prove independence from convergence.
fn external_is_convergent(name: &str) -> bool {
    intrinsic_convergence(name).unwrap_or(true)
}

/// Read an explicit LLVM convergence requirement from its upstream typed attribute.
/// Reject malformed forms before inference so they cannot silently weaken protection.
fn explicit_convergence(attrs: Option<&LlvmAttributesAttr>) -> Result<bool, String> {
    let Some((_, value)) =
        attrs.and_then(|attrs| attrs.iter().find(|(name, _)| *name == "convergent"))
    else {
        return Ok(false);
    };
    match value {
        LlvmAttrValue::Unit => Ok(true),
        _ => Err("LLVM function/call `convergent` requires a unit attribute".into()),
    }
}

/// Analyze the final LLVM module without modifying the IR or serializer state.
/// Reverse call edges propagate seeds through recursion independently of emission order.
pub(super) fn analyze_convergence<'a>(
    ctx: &Context,
    module: &ModuleOp,
    device_externs: impl IntoIterator<Item = &'a DeviceExternDecl>,
) -> Result<ConvergenceAnalysis, String> {
    let functions: Vec<_> = module
        .get_operation()
        .deref(ctx)
        .regions()
        .flat_map(|region| region.deref(ctx).iter(ctx))
        .flat_map(|block| block.deref(ctx).iter(ctx))
        .filter_map(|operation| Operation::get_op::<ops::FuncOp>(operation, ctx))
        .collect();
    let mut requirements = FxHashMap::default();
    // Index every definition before visiting calls, including callers that
    // precede their callees and components containing recursive calls.
    for func in &functions {
        let name = exported_function_name(func.get_symbol_name(ctx).as_ref());
        let defined = func.get_operation().deref(ctx).regions().count() != 0;
        let explicit = explicit_convergence(func.get_attr_llvm_func_attrs(ctx).as_deref())?;
        let intrinsic = crate::intrinsics::intrinsic_convergence(ctx, func.get_operation())?;
        requirements.insert(
            name.clone(),
            explicit
                || (!defined
                    && intrinsic
                        .map_or_else(|| external_is_convergent(&name), |fact| fact.convergent)),
        );
    }
    for decl in device_externs {
        let requirement = decl.attrs.is_convergent || external_is_convergent(&decl.export_name);
        *requirements.entry(decl.export_name.clone()).or_default() |= requirement;
    }
    let mut callers = FxHashMap::default();
    for func in &functions {
        let name = exported_function_name(func.get_symbol_name(ctx).as_ref());
        visit_convergence(
            ctx,
            func.get_operation(),
            &name,
            &mut requirements,
            &mut callers,
        )?;
    }
    let mut pending: Vec<_> = requirements
        .iter()
        .filter_map(|(name, &convergent)| convergent.then_some(name.clone()))
        .collect();
    while let Some(callee) = pending.pop() {
        if let Some(parents) = callers.get(&callee) {
            for parent in parents {
                let requirement = requirements.get_mut(parent).expect("indexed caller");
                if !*requirement {
                    *requirement = true;
                    pending.push(parent.clone());
                }
            }
        }
    }
    Ok(ConvergenceAnalysis {
        functions: requirements,
    })
}

/// Collect call edges and seed requirements from explicit call attributes and asm.
/// Visit nested regions without altering IR; explicit call annotations do not taint callees.
fn visit_convergence(
    ctx: &Context,
    operation: Ptr<Operation>,
    function: &str,
    requirements: &mut FxHashMap<String, bool>,
    callers: &mut FxHashMap<String, Vec<String>>,
) -> Result<(), String> {
    if let Some(call) = Operation::get_op::<ops::CallOp>(operation, ctx) {
        if explicit_convergence(call.get_attr_llvm_call_attrs(ctx).as_deref())? {
            *requirements.get_mut(function).expect("indexed caller") = true;
        }
        match call.callee(ctx) {
            CallOpCallable::Direct(name) => {
                let callee = exported_function_name(name.as_ref());
                requirements
                    .entry(callee.clone())
                    .or_insert_with(|| external_is_convergent(&callee));
                callers
                    .entry(callee)
                    .or_default()
                    .push(function.to_string());
            }
            CallOpCallable::Indirect(_) => {
                *requirements.get_mut(function).expect("indexed caller") = true;
            }
        }
    } else if let Some(asm) = Operation::get_op::<ops::InlineAsmOp>(operation, ctx)
        && ops::inline_asm_convergence(ctx, &asm)?
    {
        *requirements.get_mut(function).expect("indexed function") = true;
    }
    for region in operation.deref(ctx).regions() {
        for block in region.deref(ctx).iter(ctx) {
            for child in block.deref(ctx).iter(ctx) {
                visit_convergence(ctx, child, function, requirements, callers)?;
            }
        }
    }
    Ok(())
}
