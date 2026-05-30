/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Non-bulk asynchronous copy (`cp.async.*`) intrinsic conversion.
//!
//! | Operation                  | Lowering        | PTX                       |
//! |----------------------------|-----------------|---------------------------|
//! | `CpAsyncCaShared16`        | LLVM intrinsic  | `cp.async.ca.shared.global` |
//! | `CpAsyncCommitGroup`       | Inline PTX      | `cp.async.commit_group`   |
//! | `CpAsyncWaitGroup`         | Inline PTX      | `cp.async.wait_group N`   |
//!
//! The copy uses the LLVM NVVM intrinsic
//! `llvm.nvvm.cp.async.ca.shared.global.16(ptr addrspace(3), ptr addrspace(1))`
//! so the backend handles the `cvta` to the shared/global windows. The group
//! sync ops use inline PTX (no immarg-friendly LLVM intrinsic is needed and the
//! PTX is trivial).

use crate::convert::intrinsics::common::*;
use crate::helpers;
use dialect_llvm::ops as llvm;
use dialect_llvm::types as llvm_types;
use pliron::builtin::op_interfaces::CallOpCallable;
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;

/// Convert `cp_async_ca_shared_global_16` via the NVVM LLVM intrinsic.
///
/// Operands (from the importer): `[dst, src]`.
/// - `dst` is cast to `addrspace(3)` (shared)
/// - `src` is cast to `addrspace(1)` (global)
pub(crate) fn convert_copy_ca_16(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let void_ty = llvm_types::VoidType::get(ctx);
    let smem_ptr_ty = llvm_types::PointerType::get(ctx, 3);
    let global_ptr_ty = llvm_types::PointerType::get(ctx, 1);

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 2 {
        return pliron::input_err_noloc!(
            "cp.async.ca.shared.global.16 requires 2 operands, got {}",
            operands.len()
        );
    }

    let dst_casted = cast_to_shared_addrspace(ctx, rewriter, operands[0]);
    let src_casted = cast_to_global_addrspace(ctx, rewriter, operands[1]);

    let arg_types: Vec<Ptr<pliron::r#type::TypeObj>> =
        vec![smem_ptr_ty.into(), global_ptr_ty.into()];

    let intrinsic_name = "llvm_nvvm_cp_async_ca_shared_global_16";
    let func_ty = llvm_types::FuncType::get(ctx, void_ty.into(), arg_types, false);

    let parent_block = op.deref(ctx).get_parent_block().unwrap();
    helpers::ensure_intrinsic_declared(ctx, parent_block, intrinsic_name, func_ty)
        .map_err(|e| pliron::input_error_noloc!("{}", e))?;

    let call_args = vec![dst_casted, src_casted];
    let sym_name: pliron::identifier::Identifier = intrinsic_name.try_into().unwrap();
    let callee = CallOpCallable::Direct(sym_name);
    let llvm_call = llvm::CallOp::new(ctx, callee, func_ty, call_args);
    rewriter.insert_operation(ctx, llvm_call.get_operation());
    rewriter.erase_operation(ctx, op);

    Ok(())
}

/// Convert `cp_async_commit_group` to inline PTX.
pub(crate) fn convert_commit_group(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let void_ty = llvm_types::VoidType::get(ctx);
    inline_asm_convergent(
        ctx,
        rewriter,
        void_ty.into(),
        vec![],
        "cp.async.commit_group;",
        "~{memory}",
    );
    // Erase the original nvvm op (the inline asm replaces it); otherwise it
    // leaks to the LLVM export as an "Unknown op" comment, like wait_group.
    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// Convert `cp_async_wait_group` to inline PTX. `n` is an immediate operand.
pub(crate) fn convert_wait_group(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let void_ty = llvm_types::VoidType::get(ctx);
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    let n = operands
        .first()
        .copied()
        .unwrap_or_else(|| create_i32_const(ctx, rewriter, 0));

    inline_asm_convergent(
        ctx,
        rewriter,
        void_ty.into(),
        vec![n],
        "cp.async.wait_group $0;",
        "n,~{memory}",
    );
    rewriter.erase_operation(ctx, op);
    Ok(())
}
