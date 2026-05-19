// LOCAL EXPERIMENT for studio-vaai cloth-solver-cuda — DO NOT UPSTREAM.

use super::super::helpers::emit_goto;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::values::ValueMap;
use dialect_nvvm::ops::AtomicAddGlobalV4F32Op;
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use rustc_public::mir;

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
