// LOCAL EXPERIMENT for studio-vaai cloth-solver-cuda — DO NOT UPSTREAM.

use super::super::helpers::{emit_goto, emit_store_result_and_goto};
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::values::ValueMap;
use dialect_nvvm::ops::{AtomicAddGlobalV4F32Op, LdGlobalV4F32Op, StGlobalV4F32Op};
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::FP32Type;
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::value::Value;
use rustc_public::mir;

/// `cuda_device::vec::st_global_v4_f32(p, a, b, c, d)` →
/// `nvvm.st_global_v4_f32(p, a, b, c, d)`.
pub fn emit_st_global_v4_f32(
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
    if args.len() != 5 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "st_global_v4_f32 expects 5 arguments (p, a, b, c, d), got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;
    let mut operands = Vec::with_capacity(5);
    for arg in args {
        let (v, last_op_after) = rvalue::translate_operand(
            ctx,
            body,
            arg,
            value_map,
            block_ptr,
            last_op,
            loc.clone(),
        )?;
        operands.push(v);
        last_op = last_op_after;
    }

    let st_op = Operation::new(
        ctx,
        StGlobalV4F32Op::get_concrete_op_info(),
        vec![],
        operands,
        vec![],
        0,
    );
    st_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        st_op.insert_after(ctx, prev);
    } else {
        st_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        Ok(emit_goto(ctx, *target_idx, st_op, block_map, loc))
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported("st_global_v4_f32 call without target block".to_string())
        )
    }
}

/// `cuda_device::vec::ld_global_v4_f32(p) -> [f32; 4]` →
/// `nvvm.ld_global_v4_f32(p)` producing 4 f32 results, then bundled into a
/// `[f32; 4]` MIR array stored to the destination Place.
pub fn emit_ld_global_v4_f32(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
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
                "ld_global_v4_f32 expects 1 argument (p), got {}",
                args.len()
            ))
        );
    }

    let (ptr_val, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;
    let last_op = last_op_after;

    let f32_ty = FP32Type::get(ctx);
    let result_types = (0..4).map(|_| f32_ty.into()).collect();

    let ld_op = Operation::new(
        ctx,
        LdGlobalV4F32Op::get_concrete_op_info(),
        result_types,
        vec![ptr_val],
        vec![],
        0,
    );
    ld_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        ld_op.insert_after(ctx, prev);
    } else {
        ld_op.insert_at_front(block_ptr, ctx);
    }

    let results: Vec<Value> = (0..4).map(|i| ld_op.deref(ctx).get_result(i)).collect();

    let array_ty = dialect_mir::types::MirArrayType::get(ctx, f32_ty.into(), 4);
    let array_op = Operation::new(
        ctx,
        dialect_mir::ops::MirConstructArrayOp::get_concrete_op_info(),
        vec![array_ty.into()],
        results,
        vec![],
        0,
    );
    array_op.deref_mut(ctx).set_loc(loc.clone());
    array_op.insert_after(ctx, ld_op);

    let array_result = array_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        array_result,
        target,
        block_ptr,
        array_op,
        value_map,
        block_map,
        loc,
        "ld_global_v4_f32 call without target block",
    )
}

/// `cuda_device::vec::atomic_add_global_v4_f32(p, a, b, c, d)` →
/// `nvvm.atomic_add_global_v4_f32(p, a, b, c, d)`. Lowers to
/// `red.global.add.v4.f32` (SASS `REDG.E.ADD.F32x4`).
pub fn emit_atomic_add_global_v4_f32(
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
    if args.len() != 5 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "atomic_add_global_v4_f32 expects 5 arguments (p, a, b, c, d), got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;
    let mut operands = Vec::with_capacity(5);
    for arg in args {
        let (v, last_op_after) = rvalue::translate_operand(
            ctx,
            body,
            arg,
            value_map,
            block_ptr,
            last_op,
            loc.clone(),
        )?;
        operands.push(v);
        last_op = last_op_after;
    }

    let red_op = Operation::new(
        ctx,
        AtomicAddGlobalV4F32Op::get_concrete_op_info(),
        vec![],
        operands,
        vec![],
        0,
    );
    red_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        red_op.insert_after(ctx, prev);
    } else {
        red_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        Ok(emit_goto(ctx, *target_idx, red_op, block_map, loc))
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported(
                "atomic_add_global_v4_f32 call without target block".to_string(),
            )
        )
    }
}
