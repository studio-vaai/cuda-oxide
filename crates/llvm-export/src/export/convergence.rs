/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Derive convergence from operations and propagate it from callees to callers.

use pliron::{
    builtin::{
        op_interfaces::{CallOpCallable, CallOpInterface, SymbolOpInterface},
        ops::ModuleOp,
    },
    context::Ptr,
    linked_list::ContainsLinkedList,
    op::Op,
    operation::Operation,
};
use rustc_hash::FxHashMap;

use super::{function::exported_function_name, state::ModuleExportState};
use crate::ops;

/// Only intrinsic families with no inter-thread convergence semantics qualify.
/// An unknown intrinsic, external declaration, or indirect call remains
/// conservative; read-only/pure memory effects alone do not prove convergence.
fn known_nonconvergent_intrinsic(name: &str) -> bool {
    matches!(name, "llvm.trap" | "llvm.debugtrap" | "llvm.assume")
        || [
            "llvm.abs.",
            "llvm.bswap.",
            "llvm.bitreverse.",
            "llvm.ctlz.",
            "llvm.cttz.",
            "llvm.ctpop.",
            "llvm.fshl.",
            "llvm.fshr.",
            "llvm.fabs.",
            "llvm.copysign.",
            "llvm.sqrt.",
            "llvm.pow.",
            "llvm.powi.",
            "llvm.sin.",
            "llvm.cos.",
            "llvm.exp.",
            "llvm.exp2.",
            "llvm.log.",
            "llvm.log2.",
            "llvm.log10.",
            "llvm.floor.",
            "llvm.ceil.",
            "llvm.trunc.",
            "llvm.rint.",
            "llvm.nearbyint.",
            "llvm.round.",
            "llvm.roundeven.",
            "llvm.fma.",
            "llvm.fmuladd.",
            "llvm.minnum.",
            "llvm.maxnum.",
            "llvm.minimum.",
            "llvm.maximum.",
            "llvm.sadd.with.overflow.",
            "llvm.uadd.with.overflow.",
            "llvm.ssub.with.overflow.",
            "llvm.usub.with.overflow.",
            "llvm.smul.with.overflow.",
            "llvm.umul.with.overflow.",
            "llvm.sadd.sat.",
            "llvm.uadd.sat.",
            "llvm.ssub.sat.",
            "llvm.usub.sat.",
            "llvm.memcpy.",
            "llvm.memmove.",
            "llvm.memset.",
            "llvm.lifetime.start.",
            "llvm.lifetime.end.",
            "llvm.nvvm.read.ptx.sreg.",
        ]
        .iter()
        .any(|prefix| name.starts_with(prefix))
}

impl ModuleExportState<'_> {
    pub(super) fn function_is_convergent(&self, name: &str) -> bool {
        self.convergent_functions.contains(name)
            || (!self.function_definitions.contains(name) && !known_nonconvergent_intrinsic(name))
    }

    /// Analyze the complete exported call graph before emitting any definition.
    /// Reverse edges make recursive components order-independent: a convergent
    /// operation seeds its containing function, then every transitive caller.
    pub(super) fn infer_convergence(&mut self, module: &ModuleOp) {
        let mut callers: FxHashMap<String, Vec<String>> = FxHashMap::default();
        for decl in self.device_externs.values() {
            if decl.attrs.is_convergent {
                self.convergent_functions.insert(decl.export_name.clone());
            }
        }
        for region in module.get_operation().deref(self.ctx).regions() {
            for block in region.deref(self.ctx).iter(self.ctx) {
                for operation in block.deref(self.ctx).iter(self.ctx) {
                    let Some(func) = Operation::get_op::<ops::FuncOp>(operation, self.ctx) else {
                        continue;
                    };
                    let name = exported_function_name(func.get_symbol_name(self.ctx).as_ref());
                    if !self.function_definitions.contains(&name) {
                        if Self::is_convergent_intrinsic(&name)
                            || !known_nonconvergent_intrinsic(&name)
                        {
                            self.convergent_functions.insert(name);
                        }
                        continue;
                    }
                    self.visit_convergence(operation, &name, &mut callers);
                }
            }
        }
        let mut pending: Vec<_> = self.convergent_functions.iter().cloned().collect();
        while let Some(callee) = pending.pop() {
            if let Some(parents) = callers.get(&callee) {
                for parent in parents {
                    if self.convergent_functions.insert(parent.clone()) {
                        pending.push(parent.clone());
                    }
                }
            }
        }
    }

    fn visit_convergence(
        &mut self,
        operation: Ptr<Operation>,
        function: &str,
        callers: &mut FxHashMap<String, Vec<String>>,
    ) {
        if let Some(call) = Operation::get_op::<ops::CallOp>(operation, self.ctx) {
            match call.callee(self.ctx) {
                CallOpCallable::Direct(name) => {
                    let callee = exported_function_name(name.as_ref());
                    callers
                        .entry(callee.clone())
                        .or_default()
                        .push(function.to_string());
                    if !self.function_definitions.contains(&callee)
                        && (Self::is_convergent_intrinsic(&callee)
                            || !known_nonconvergent_intrinsic(&callee))
                    {
                        self.convergent_functions.insert(callee);
                    }
                }
                CallOpCallable::Indirect(_) => {
                    self.convergent_functions.insert(function.to_string());
                }
            }
        } else if let Some(asm) = Operation::get_op::<ops::InlineAsmOp>(operation, self.ctx)
            && inline_asm_is_convergent(self.ctx, &asm)
        {
            self.convergent_functions.insert(function.to_string());
        }
        let children: Vec<_> = operation
            .deref(self.ctx)
            .regions()
            .flat_map(|region| region.deref(self.ctx).iter(self.ctx))
            .flat_map(|block| block.deref(self.ctx).iter(self.ctx))
            .collect();
        for child in children {
            self.visit_convergence(child, function, callers);
        }
    }
}

pub(super) fn inline_asm_is_convergent(
    ctx: &pliron::context::Context,
    asm: &ops::InlineAsmOp,
) -> bool {
    match ops::asm_kind_opt(ctx, asm) {
        Some(ops::AsmKind::Convergent | ops::AsmKind::ConvergentPure) => true,
        Some(ops::AsmKind::SideEffect | ops::AsmKind::Pure) => false,
        None => asm
            .get_attr_inline_asm_convergent(ctx)
            .map(|attr| bool::from((*attr).clone()))
            .unwrap_or(false),
    }
}
