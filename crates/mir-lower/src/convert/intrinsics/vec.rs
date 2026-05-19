// LOCAL EXPERIMENT for studio-vaai cloth-solver-cuda — DO NOT UPSTREAM.

use dialect_llvm::ops as llvm;
use dialect_llvm::types as llvm_types;
use pliron::builtin::types::FP32Type;
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::TypeObj;

/// Convert `nvvm.st_global_v4_f32(p, a, b, c, d)` → inline PTX.
///
/// PTX: `st.global.v4.f32 [p], {a, b, c, d};`
pub(crate) fn convert_st_global_v4_f32(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 5 {
        return pliron::input_err_noloc!("st_global_v4_f32 requires 5 operands");
    }

    let void_ty = llvm_types::VoidType::get(ctx);
    // No `~{memory}` clobber — pure store. The `sideeffect` flag from `new()`
    // prevents elimination; `~{memory}` would force LLVM to spill/reload all
    // live memory across each call site, defeating optimization.
    let inline_asm = llvm::InlineAsmOp::new(
        ctx,
        void_ty.into(),
        operands,
        "st.global.v4.f32 [$0], {$1, $2, $3, $4};",
        "l,f,f,f,f",
    );
    rewriter.insert_operation(ctx, inline_asm.get_operation());
    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// Convert `nvvm.ld_global_v4_f32(p) -> 4xf32` → inline PTX.
///
/// PTX: `ld.global.v4.f32 {a, b, c, d}, [p];`
///
/// Inline asm returns an LLVM struct {f32, f32, f32, f32}; we extract each
/// field and bind to the 4 op results.
pub(crate) fn convert_ld_global_v4_f32(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 1 {
        return pliron::input_err_noloc!("ld_global_v4_f32 requires 1 operand (ptr)");
    }

    let f32_ty = FP32Type::get(ctx);
    let field_types: Vec<Ptr<TypeObj>> = (0..4).map(|_| f32_ty.into()).collect();
    let struct_ty = llvm_types::StructType::get_unnamed(ctx, field_types);

    let inline_asm = llvm::InlineAsmOp::new(
        ctx,
        struct_ty.into(),
        operands,
        "ld.global.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
    );
    let asm_op = inline_asm.get_operation();
    rewriter.insert_operation(ctx, asm_op);

    let struct_result = asm_op.deref(ctx).get_result(0);
    let mut extracted_values = Vec::with_capacity(4);
    for i in 0..4u32 {
        let extract_op = llvm::ExtractValueOp::new(ctx, struct_result, vec![i])
            .map_err(|e| pliron::input_error_noloc!("{}", e))?;
        rewriter.insert_operation(ctx, extract_op.get_operation());
        let field_val = extract_op.get_operation().deref(ctx).get_result(0);
        extracted_values.push(field_val);
    }
    rewriter.replace_operation_with_values(ctx, op, extracted_values);

    Ok(())
}

/// Convert `nvvm.atomic_add_global_v4_f32(p, a, b, c, d)` → inline PTX.
///
/// PTX: `red.global.add.v4.f32 [p], {a, b, c, d};`
///
/// SASS: `REDG.E.ADD.F32x4`. Native single-instruction vector atomic-add on
/// Blackwell (sm_9.0+). No previous-value output (use `atom.global.add.v4.f32`
/// with `=f,=f,=f,=f` outputs if needed; not provided here since the cloth
/// solver doesnt use per-element previous values).
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
