/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Non-bulk asynchronous copy (`cp.async.*`) intrinsics.
//!
//! Per-element LDGSTS copies (global→shared) used for software-pipelined
//! prefetch, distinct from the descriptor-based TMA bulk copies in [`super::tma`].

use super::super::helpers::emit_goto;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::values::ValueMap;
use dialect_nvvm::ops::{CpAsyncCaShared16Op, CpAsyncCommitGroupOp, CpAsyncWaitGroupOp};
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use rustc_public::mir;

/// Emit `cp_async_ca_shared_global_16`: async 16-byte global→shared copy.
///
/// Args:
/// - `args[0]`: `*mut u8`   — destination in shared memory
/// - `args[1]`: `*const u8` — source in global memory
///
/// Returns: void
pub fn emit_cp_async_ca_shared_global_16(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 2 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "cp_async_ca_shared_global_16 expects 2 arguments, got {}",
                args.len()
            ))
        );
    }

    let mut operands = Vec::new();
    let mut last_op = prev_op;

    // arg[0]: dst (shared memory pointer)
    let (dst, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    operands.push(dst);
    last_op = last_op_after;

    // arg[1]: src (global memory pointer)
    let (src, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    operands.push(src);
    last_op = last_op_after;

    // Create the copy operation (void return, operands: dst, src)
    let copy_op = Operation::new(
        ctx,
        CpAsyncCaShared16Op::get_concrete_op_info(),
        vec![], // No results
        operands,
        vec![],
        0,
    );
    copy_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        copy_op.insert_after(ctx, prev);
    } else {
        copy_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, copy_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported(
                "cp_async_ca_shared_global_16 call without target block".to_string()
            )
        )
    }
}

/// Emit `cp_async_commit_group`: commit pending async copies into a group.
///
/// Args: none. Returns: void.
pub fn emit_cp_async_commit_group(
    ctx: &mut Context,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if !args.is_empty() {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "cp_async_commit_group expects 0 arguments, got {}",
                args.len()
            ))
        );
    }

    let commit_op = Operation::new(
        ctx,
        CpAsyncCommitGroupOp::get_concrete_op_info(),
        vec![], // No results
        vec![], // No operands
        vec![],
        0,
    );
    commit_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = prev_op {
        commit_op.insert_after(ctx, prev);
    } else {
        commit_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, commit_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported("cp_async_commit_group call without target block".to_string())
        )
    }
}

/// Emit `cp_async_wait_group`: wait for async copy groups to drain.
///
/// Args:
/// - `args[0]`: `u32` — max pending groups (0 = wait for all)
///
/// Returns: void
pub fn emit_cp_async_wait_group(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 1 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "cp_async_wait_group expects 1 argument, got {}",
                args.len()
            ))
        );
    }

    let (count, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    let wait_op = Operation::new(
        ctx,
        CpAsyncWaitGroupOp::get_concrete_op_info(),
        vec![],      // No results
        vec![count], // Operand: count
        vec![],
        0,
    );
    wait_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        wait_op.insert_after(ctx, prev);
    } else {
        wait_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, wait_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported("cp_async_wait_group call without target block".to_string())
        )
    }
}
