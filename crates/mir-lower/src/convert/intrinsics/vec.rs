// LOCAL EXPERIMENT for studio-vaai cloth-solver-cuda — DO NOT UPSTREAM.

use dialect_llvm::ops as llvm;
use dialect_llvm::types as llvm_types;
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;

/// Convert `nvvm.atomic_add_global_v4_f32(p, a, b, c, d)` → inline PTX.
///
/// PTX: `red.global.add.v4.f32 [p], {a, b, c, d};`
///
/// SASS: `REDG.E.ADD.F32x4`. Native single-instruction vector atomic-add on
/// Blackwell (sm_9.0+). No previous-value output (use `atom.global.add.v4.f32`
/// with `=f,=f,=f,=f` outputs if needed; not provided here since the cloth
/// solver doesn't use per-element previous values).
pub(crate) fn convert_atomic_add_global_v4_f32(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 5 {
        return pliron::input_err_noloc!("atomic_add_global_v4_f32 requires 5 operands");
    }

    let void_ty = llvm_types::VoidType::get(ctx);
    let inline_asm = llvm::InlineAsmOp::new(
        ctx,
        void_ty.into(),
        operands,
        "red.global.add.v4.f32 [$0], {$1, $2, $3, $4};",
        "l,f,f,f,f",
    );
    rewriter.insert_operation(ctx, inline_asm.get_operation());
    rewriter.erase_operation(ctx, op);
    Ok(())
}
